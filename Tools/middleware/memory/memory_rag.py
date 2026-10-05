import asyncio
import json
import math
import os
import sqlite3
import time
from asyncio import Lock
from datetime import datetime

from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
import redis.asyncio as redis
from typing import Optional, List, Tuple, Literal, Union, Any

from aio_pika.abc import AbstractIncomingMessage
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
from langchain_core.runnables.retry import ExponentialJitterParams
from pydantic import Field, BaseModel, field_validator

from Tools.email import send_error_email
from Tools.middleware.memory.customize_sqlite_vec import CustomizeSQLiteVec
from load_config.config import config, get_first_db_by_conf, is_absolute_path, ROOT_BASE_DIR_PATH
from SPO.memory import MemoryFragments, SummaryMemoryAi, MemoryFragmentsMetadata, UserProfile
from Tools.log_settings import LogSetting

logger = LogSetting.create(__name__)
rag_conf = config.get('databases').get('rag')
if not rag_conf:
    raise ValueError("rag数据库配置错误")
rag_db, rag_args = get_first_db_by_conf(rag_conf)
if not rag_db or not rag_args:
    logger.error("rag数据库配置错误")
    raise ValueError("rag数据库配置错误")
_rabbitmq_config = config.get('AMQP').get('rabbitmq')
if not _rabbitmq_config:
    raise ValueError("AMQP配置未找到, 请先配置对应的AMQP配置")

# aio-pika 参数字典
_rabbit_params = dict(
    host=_rabbitmq_config.get('host'),
    virtualhost=_rabbitmq_config.get('virtual_host', '/'),
    login=_rabbitmq_config.get('username'),
    password=_rabbitmq_config.get('password'),
    heartbeat=_rabbitmq_config.get('heartbeat', 600),
    # connection_attempts × retry_delay 反用为单次连接超时（初始连接失败为 fail-fast）
    timeout=_rabbitmq_config.get('connection_attempts', 3) * _rabbitmq_config.get('retry_delay', 5),
    # retry_delay 对应断线后的自动重连间隔
    reconnect_interval=_rabbitmq_config.get('retry_delay', 5),
)
rabbit_conn = None
channel = None
_rabbit_lock = Lock()
_rag_error_exchange = None

# 定义对应的队列列名
summary_queue_name = "unclassified_fragments"
split_queue_name = "uncut_text"
# 定义对应的 routing_key
summary_routing_key = "unclassified_fragments_data"
split_routing_key = "uncut_text_data"

# (routing_key, user_id)
_nack_fail_counts: dict = dict()
# nack requeue 前的退避等待（秒）：避免失败消息高频循环
_RETRY_SLEEP_SECONDS = 3


class ErrorFragmentsData(BaseModel):
    type: Literal['spliter', 'summary'] = Field(default='spliter', description='错误引发的源头')
    user_id: str = Field(default='-1', description="用户ID")
    # List[Any]：生产端手动 dump 消息为 dict（防止 langchain 嵌套序列化丢失 tool_calls/tool_call_id字段）
    text: Union[str, List[Any]] = Field(default='', description="未区分的文本（summary）或原始消息列表（spliter）")
    last_summary_id: Optional[int] = Field(default=None, description='增量总结游标，供消费者恢复进度')

    # 将消息还原为对应的 BaseMessage 子类
    @field_validator('text', mode='before')
    @classmethod
    def _restore_messages(cls, v):
        """
        pydantic 反序列化 List[BaseMessage] 按其中的 type 字段还原类型
        """
        if isinstance(v, list) and v and all(isinstance(item, dict) for item in v):
            msg_type_map = {
                'human': HumanMessage,
                'ai': AIMessage,
                'tool': ToolMessage,
                'system': SystemMessage,
            }
            restored = []
            for item in v:
                msg_type = item.get('type', '')
                msg_cls = msg_type_map.get(msg_type, BaseMessage)
                fields = {k: val for k, val in item.items() if k != 'type'}
                try:
                    restored.append(msg_cls(**fields))
                except Exception as e:
                    logger.warning(f"恢复消息 {item} 时出错: {e}")
                    # 未知/异常字段时兜底为基类，并补回 type 字段（BaseMessage 的 type 必填）
                    restored.append(BaseMessage(**{**fields, 'type': msg_type}))
            return restored
        return v


def _get_memory_instances() -> Tuple["MemoryFragmentsAiSpliter", "FragmentsMemoryRAG"]:
    # 延迟导入避免循环依赖（agent.main_agent -> time_memory -> memory_rag）
    from agent.main_agent import main_middlewares
    from Tools.middleware.memory.time_memory import BalancedMultiDimensionMemory
    for middleware in main_middlewares:
        if isinstance(middleware, BalancedMultiDimensionMemory):
            return middleware.memory_spliter, middleware.memory_fragments_rag
    raise RuntimeError(
        "main_middlewares 中未找到 BalancedMultiDimensionMemory 实例，请检查中间件配置"
    )


async def _consumer_process_error_message(message: AbstractIncomingMessage):
    routing_key = message.routing_key
    data = None
    try:
        data = ErrorFragmentsData.model_validate_json(message.body)
        if routing_key == split_routing_key:
            # 通过 MemoryFragmentsAiSpliter.split_llm 重新切分，恢复未完成的记忆分片
            await _recover_split(data)
        elif routing_key == summary_routing_key:
            # 通过 FragmentsMemoryRAG._summary_llm 重新总结，恢复未完成的用户画像
            await _recover_summary(data)
        # 处理成功后才确认消息，确保消息被正确处理
        await message.ack()
    except Exception as e:
        # nack requeue 会立即重投,限频日志：
        fail_key = (routing_key, data.user_id if data else '-1')
        fail_count = _nack_fail_counts.get(fail_key, 0) + 1
        _nack_fail_counts[fail_key] = fail_count
        if fail_count % 100 == 0:
            logger.warning(f"处理错误消息时出错（已连续失败 {fail_count} 次）: {e}")
        # 出错后，将消息返回到错误队列，进行重试。
        # 不将消息放到死信队列中，确保消息处理，而不是丢弃
        # 超时通知：消息入队超过1天仍未处理成功则邮件通知（requeue 不会重置消息 timestamp）
        elapsed = time.time() - message.timestamp.timestamp() if message.timestamp else 0
        if elapsed >= 86400:
            try:
                await _notify_stale_message(data.user_id if data else '-1', routing_key, elapsed)
            except Exception as notify_e:
                # 通知失败不能阻塞重试（否则消息会被卡住不 requeue）
                logger.error(f"长时间未处理通知发送失败: {notify_e}")
        # 退避：sleep 后 nack，避免 requeue 立即重投造成高频循环
        await asyncio.sleep(_RETRY_SLEEP_SECONDS)
        await message.nack()


async def _recover_split(data: ErrorFragmentsData) -> None:
    """恢复 split 队列中未完成的消息切分：LLM 重新切分 -> 落 RAG，Redis 游标续接"""
    spliter, rag = _get_memory_instances()
    if not spliter.split_llm:
        # 初始化
        spliter.create_agent()
    messages = data.text
    if not messages or isinstance(messages, str) or (isinstance(messages, list) and isinstance(messages[0], str)):
        # 防止出现处理过后的消息，记录后跳过
        logger.warning(f"spliter 队列收到旧格式消息，无法恢复切分，已跳过: user_id={data.user_id}")
        return
    # 与 asplit_text 相同的切分流程（不调 asplit_text：若其失败会再次发布新消息，形成循环）
    spliter.dialogue_theme_by_user.setdefault(data.user_id, '')
    text = []
    content_text = []
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            role = 'ai'
            content = message.content
            if not content and message.tool_calls:
                content = '（调用了工具 ' + ', '.join(tc['name'] for tc in message.tool_calls) + '，无文本内容）'
        elif isinstance(message, ToolMessage):
            role = 'tool'
            content = message.content if isinstance(message.content, str) else str(message.content)
        elif isinstance(message, HumanMessage):
            role = 'user'
            content = message.content
        else:
            role = 'other'
            content = message.content if message.content else '（非对话消息）'
        if not content:
            content = '（消息为空）'
        if role == 'tool' and len(content) > 100:
            content = content[:100] + '...'
        text.append(f'对应的消息索引{i}，{role}:{content}')
        content_text.append(content)
    prompt = "请将以下文本按照不同主题进行切分区域：\n" + "\n".join(text)
    result = await spliter.split_llm.ainvoke([
        SystemMessage(content=spliter.prompt),
        HumanMessage(content=prompt)
    ])
    spliter.dialogue_theme_by_user[data.user_id] = result.current_theme
    memory_fragments = spliter.split_text(result.theme_num, result.config, content_text, messages)
    if not memory_fragments:
        return
    # 与 atext_to_document 相同的落库逻辑：Redis 游标续接（INCRBY 原子预留）+ 构造 Document + 写入 RAG
    fragment_base = await spliter.fragment_id_base(data.user_id)
    if fragment_base > int(await spliter.redis_con.get(f"memory_fragments:{data.user_id}") or '0'):
        await spliter.redis_con.set(f"memory_fragments:{data.user_id}", str(fragment_base))
    last_id = await spliter.redis_con.incrby(f"memory_fragments:{data.user_id}", len(memory_fragments))
    current_index = last_id - len(memory_fragments)
    documents = []
    for fragment in memory_fragments:
        current_index += 1
        metadata = fragment.config.model_dump()
        metadata['user_id'] = data.user_id
        metadata['id'] = current_index
        documents.append(Document(
            id=f"{data.user_id}-{current_index}",
            page_content=fragment.content,
            metadata=metadata,
        ))
    await rag.add(documents)
    logger.info(f"消费者恢复切分完成: user_id={data.user_id}, 片段数={len(documents)}")


async def _recover_summary(data: ErrorFragmentsData) -> None:
    """恢复 summary 队列中未完成的画像总结：LLM 总结 -> 合并旧画像 -> 写回 store（游标推进）"""
    from agent.main_agent import graph
    _, rag = _get_memory_instances()
    if not rag.summary_llm:
        # 服务重启后懒初始化（复用同一套模型配置，不开新的 init_chat_model）
        summary_conf = config['model']['summary']
        await rag.create_summary_agent(summary_conf['name'], **summary_conf['kwargs'])
    profile = await rag.summary_llm.ainvoke([
        SystemMessage(content=rag.prompt),
        HumanMessage(content=f"目前的新增对话片段如下：\n{data.text}"),
    ])
    profile_json = profile.model_dump(mode='json')
    profile_json['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"用户画像（消费者恢复总结）：{profile_json}")
    store = graph.store
    if store is None:
        logger.warning("未配置 langgraph store，跳过长期记忆存储")
        return
    # 与 time_memory._memory_slice 的落库流程一致：合并旧画像 + 推进游标
    from Tools.middleware.memory.time_memory import BalancedMultiDimensionMemory
    existing = {}
    try:
        item = await store.aget(('long-term', 'user_profile',), data.user_id)
        if item is not None:
            existing = item.value.get('user_profile', {}) or {}
    except Exception as e:
        logger.warning(f"读取用户画像失败，按空画像处理: {e}")
    merged = BalancedMultiDimensionMemory.merge_user_profile(existing, profile_json)
    merged['last_summarized_id'] = data.last_summary_id if data.last_summary_id is not None \
        else existing.get('last_summarized_id', 0)
    if merged.get('user_person'):
        await store.aput(('long-term', 'user_person',), data.user_id, {'user_person': merged['user_person']})
    await store.aput(('long-term', 'user_profile',), data.user_id, {'user_profile': merged})
    logger.info(f"消费者恢复总结落库完成: user_id={data.user_id}")


async def _notify_stale_message(user_id: str, routing_key: str, elapsed: float) -> None:
    """错误消息入队超过1天仍未处理成功时的邮件通知（每用户每路由每天最多一次，防刷屏）"""
    spliter, _ = _get_memory_instances()
    redis_con = spliter.redis_con
    try:
        ok = await redis_con.set(f"rag_error_notify:{user_id}:{routing_key}", '1', ex=86400, nx=True)
        if not ok:
            return
    except Exception as e:
        logger.error(f"将错误通知标记设置失败: {e}")
    send_error_email(
        subject="RAG错误消息长时间未处理",
        content=f"队列 {routing_key} 中用户 {user_id} 的错误消息已超过 {int(elapsed / 3600)} 小时未处理成功，"
                f"请检查服务与LLM状态。该消息仍在队列中持续重试，不会丢失。"
    )


async def start_consumers() -> None:
    """应用启动时主动初始化 RabbitMQ 连接/声明并注册消费者（在 main.py lifespan 中调用）"""
    await _ensure_rabbit_ready()


async def _ensure_rabbit_ready():
    """
    惰性初始化 RabbitMQ：首次发布前建立连接并声明交换机/队列/绑定，
    """
    global rabbit_conn, channel, _rag_error_exchange
    if _rag_error_exchange is not None:
        return _rag_error_exchange
    async with _rabbit_lock:
        if _rag_error_exchange is not None:
            return _rag_error_exchange
        rabbit_conn = await connect_robust(**_rabbit_params)
        channel = await rabbit_conn.channel()
        exchange = await channel.declare_exchange('rag_error', ExchangeType.DIRECT, durable=True)
        for queue_name, routing_key in (
                (summary_queue_name, summary_routing_key),
                (split_queue_name, split_routing_key),
        ):
            queue = await channel.declare_queue(queue_name, durable=True)
            await queue.bind(exchange, routing_key=routing_key)
            await queue.consume(
                callback=_consumer_process_error_message,
                timeout=120,  # 120秒超时，避免无响应
            )
        _rag_error_exchange = exchange
        return exchange


_embeddings = DashScopeEmbeddings(model="text-embedding-v4")


class MemoryFragmentsAiSpliter:
    def __init__(self, model_name: str, model_kwargs: dict[str, any]):
        self.summary_name = model_name
        self.model_kwargs = model_kwargs
        self.split_llm = None
        # 优先从 config.yaml 读取 Redis 配置，未配置时使用本机默认
        redis_conf = config.get('redis', {})
        self._redis_pool = redis.ConnectionPool(
            host=redis_conf.get('host', 'localhost'),
            port=redis_conf.get('port', 6379),
            decode_responses=True,
            password=redis_conf.get('password', None),
        )
        self.redis_con = redis.Redis(connection_pool=self._redis_pool)
        self.dialogue_theme_by_user: dict[str, str] = dict()
        # 片段 id / 消息偏移的持久化高水位来源（Redis 计数器丢失后的兜底；不注入则行为与原来一致）
        self.id_source = None
        self._id_reconciled: set[str] = set()
        self._id_high_water: dict[str, int] = dict()
        self.prompt = """
        你现在是一个专业的摘要模型，你的任务是将一个长对话文本按照不同主题进行切分区域，
        并输出每个片段的主题、可能出现的类型、片段的内容等。
        注意：
            - 输出的主题，你自己决定。（但不宜过长，尽量在30字以内）
            - 只支持的类型（绝对不允许出现除了以下类型的其他类型）（一个主题允许存在多个类型）：
                - identity: 包含用户身份信息（姓名、职业等）
                - preference: 包含用户偏好、习惯
                - decision: 包含关键决策、承诺
                - fact: 包含客观事实、技术细节
                - episode: 描述某个经历或任务过程
                - chat: 一般性闲聊，长期价值低
            - 类型判定只看内容本身，不看语气：用户陈述的个人信息 / 偏好 / 约束（姓名、职业、生日、机型、
              订单号、卡尾号、作息、饮食、住址等）一律标 identity / preference / fact，
              绝不能因为夹在闲聊里就标成 chat —— chat 只用于确实没有信息量的寒暄。
            - 片段范围：必须按照主题划分区域，且不允许跨主题。
              用 start_idx 与 end_idx 两个整数表示，区间为左闭右开 [start_idx, end_idx)，即覆盖索引 start_idx ~ end_idx-1 的消息；
              如 start_idx=0、end_idx=50 表示索引 0~49 的消息；要表达单条消息，用 start_idx=0、end_idx=1
            - 每次输出要保证 主题总数 与 记忆片段配置列表 的长度一致。
            - 一个消息会对应相对应的类型并且包含消息的索引号（从0开始），
              若消息过长，将会截取字符，一般发生在tool类型的消息中，
                如：
                对应的消息索引0，ai:你好
                对应的消息索引1，user:你好
                对应的消息索引2，tool:你好
                ....
        """

    def create_agent(self):
        if not self.split_llm:
            initial_model = init_chat_model(self.summary_name, **self.model_kwargs)
            self.split_llm = initial_model.with_structured_output(SummaryMemoryAi).with_retry(
                retry_if_exception_type=(Exception,),
                wait_exponential_jitter=True,
                stop_after_attempt=3,
                exponential_jitter_params=ExponentialJitterParams(
                    initial=2.0,
                    max=10,
                    exp_base=2,
                    jitter=1.0
                ),
            )

    async def asplit_text(
            self,
            messages: list[BaseMessage],
            user_id: str,
            current_index: Optional[int] = 0
    ) -> List[MemoryFragments] | None:
        self.create_agent()
        if not self.split_llm:
            logger.error("摘要模型未初始化")
            raise ValueError("摘要模型未初始化")
        self.dialogue_theme_by_user.setdefault(user_id, '')
        messages = messages[current_index:]
        text = []
        content_text = []
        for i, message in enumerate(messages):
            # 注意：content_text 必须与 messages 一一对应。
            # 空 content 的消息（如纯工具调用的 AIMessage）也要占位，
            # 否则 text 索引与消息索引错位，LLM 返回的 scope 会切错内容。
            if isinstance(message, AIMessage):
                role = 'ai'
                content = message.content
                if not content and message.tool_calls:
                    content = '（调用了工具 ' + ', '.join(tc['name'] for tc in message.tool_calls) + '，无文本内容）'
            elif isinstance(message, ToolMessage):
                role = 'tool'
                content = message.content if isinstance(message.content, str) else str(message.content)
            elif isinstance(message, HumanMessage):
                role = 'user'
                content = message.content
            else:
                role = 'other'
                content = message.content if message.content else '（非对话消息）'
            if not content:
                content = '（消息为空）'
            if role == 'tool' and len(content) > 100:
                content = content[:100] + '...'
            text.append(f'对应的消息索引{i}，{role}:{content}')
            content_text.append(content)
        prompt = "请将以下文本按照不同主题进行切分区域：\n" + "\n".join(text)
        try:
            result = await self.split_llm.ainvoke([
                SystemMessage(content=self.prompt),
                HumanMessage(content=prompt)
            ])
            splited_texts = result
        except Exception as e:
            logger.error(f"在记忆分片时出现了异常：{e}")
            # 错误消息投递失败只记日志，邮件兜底放在 finally 保证照常发送
            try:
                exchange = await _ensure_rabbit_ready()
                await exchange.publish(
                    Message(
                        # 手动 dump 消息为 dict：langchain 嵌套序列化会丢失 tool_calls/tool_call_id
                        body=ErrorFragmentsData(
                            type='spliter',
                            user_id=user_id,
                            text=[message.model_dump(mode='json') for message in messages]
                        ).model_dump_json().encode(),
                        delivery_mode=DeliveryMode.PERSISTENT,
                        content_type='application/json',
                        timestamp=time.time(),
                    ),
                    routing_key=split_routing_key,
                    # 显式保持 pika basic_publish 的默认语义
                    mandatory=False,
                )
            except Exception as publish_e:
                logger.error(f"发送错误消息到RabbitMQ失败：{publish_e}")
            finally:
                email_result = send_error_email(subject="智能切分异常", content=f"在进行智能切分时，出现异常：{e}")
                if not email_result:
                    logger.error("发送错误邮件失败, 请查阅相关的日志信息来锁定问题")
            # 返回None，而不选择保底机制，是因为 SummaryMemoryAi 涉及到一些类型与主题，这些无法准确确定
            return None
        self.dialogue_theme_by_user[user_id] = splited_texts.current_theme
        theme_num = splited_texts.theme_num
        conf = splited_texts.config
        return self.split_text(theme_num, conf, content_text, messages)

    async def atext_to_document(
            self,
            memory_fragments: Optional[List[MemoryFragments]] = None,
            text: Optional[list[BaseMessage]] = None,
            user: Optional[str] = None,
    ) -> List[Document] | None:
        """切分结果落库为带元数据的 Document。

        双游标修复（此前片段数≠消息数时切分窗口错位，导致片段丢失/重复）：
        - memory_fragments:{user}：片段 id 基数（每片段 +1，用于文档 id 与增量归纳游标）
        - memory_msg_offset:{user}：已切分的消息数（切分窗口偏移，成功切片后推进到 len(text)）
        text 必须传"完整消息列表"（从会话开头），偏移由消息游标控制，保证全局单调。
        """
        fragment_base = await self.fragment_id_base(user)
        raw_offset = await self.redis_con.get(f"memory_msg_offset:{user}")
        msg_offset = int(raw_offset or '0')
        if raw_offset is None:
            # Redis 丢了：从片段元数据的持久高水位恢复切分进度，避免同一会话被从 0 重切出重复片段
            probed = await self._probed_ids(user)
            if probed is not None:
                msg_offset = max(msg_offset, probed[1])
        # 新会话检测：消息列表比游标短 → 游标属于上一个会话的消息流，重置为 0。
        # （偏移语义是"本消息流已切分的消息数"；同一会话内 state['messages'] 只增不减）
        if len(text or []) < msg_offset:
            msg_offset = 0
        if not memory_fragments:
            memory_fragments = await self.asplit_text(text, user, msg_offset)
        # 由 asplit_text 产出，一般是因为出现网络异常或者是API欠费等。
        if not memory_fragments:
            return None
        # 原子预留：先把计数器抬到持久高水位（Redis 丢失或落后时），再一次性 INCRBY 拿 n 个 id。
        # 多进程下每个进程各自可能做一次校正，但 INCRBY 保证各进程拿到的 id 段互不重叠。
        if fragment_base > int(await self.redis_con.get(f"memory_fragments:{user}") or '0'):
            await self.redis_con.set(f"memory_fragments:{user}", str(fragment_base))
        last_id = await self.redis_con.incrby(f"memory_fragments:{user}", len(memory_fragments))
        documents = []
        fragment_id = last_id - len(memory_fragments)
        for fragment in memory_fragments:
            fragment_id += 1
            metadata = fragment.config.model_dump()
            # 打上用户归属与片段序号：用于跨用户检索过滤（user_id）与增量归纳游标（id）
            metadata['user_id'] = user
            metadata['id'] = fragment_id
            # 文档 id 需全局唯一（sqlite 表内有 UNIQUE 约束），带 user 前缀避免多用户撞 id
            document = Document(
                id=f"{user}-{fragment_id}",
                page_content=fragment.content,
                metadata=metadata,
            )
            documents.append(document)
        if documents:
            # 最后一片记下"切到哪了"：Redis 丢失后据此恢复消息偏移（旧数据无此键 → 取 0，行为不变）
            documents[-1].metadata['msg_end'] = len(text or [])
        # 计数器已由 incrby 原子写回（值 = last_id），这里只更新进程内高水位
        self._id_high_water[user] = last_id
        # 消息偏移 = 完整消息列表长度（切分失败时不推进，下轮重试）
        await self.redis_con.set(f"memory_msg_offset:{user}", str(len(text or [])))
        return documents

    async def _probed_ids(self, user: str):
        """向持久化来源探一次 (最大片段 id, 最大已切分消息数)；未注入来源或查询失败返回 None"""
        if self.id_source is None:
            return None
        try:
            return self.id_source.max_fragment_ids(user)
        except Exception as e:
            logger.warning(f"片段高水位兜底查询失败，沿用 Redis 计数器: {e}")
            return None

    async def fragment_id_base(self, user: str) -> int:
        """片段 id 基数：Redis 快路径 + 持久高水位兜底，只抬高不回退。

        触发兜底查询的条件（其余情况零额外查询）：
        - Redis 键缺失或为 0（计数器丢失的失败信号）
        - 本进程尚未校正过该用户（覆盖"Redis 存在但落后"，如恢复了旧 RDB 快照）
        """
        raw = await self.redis_con.get(f"memory_fragments:{user}")
        base = int(raw or '0')
        if raw is None or base == 0 or user not in self._id_reconciled:
            probed = await self._probed_ids(user)
            if probed is not None:
                base = max(base, probed[0])
                self._id_reconciled.add(user)
        return max(base, self._id_high_water.get(user, 0))

    @staticmethod
    def split_text(theme_num, conf, text: list[str], messages: list[BaseMessage]) -> List[MemoryFragments]:
        if theme_num <= 0:
            logger.warning(f"长主题数量必须大于0，当前主题数量为{theme_num}")
            return list()
        if theme_num != len(conf):
            # LLM 输出不稳定：数量不一致时截断处理而不是直接抛异常，避免整轮请求失败
            logger.error(f"长主题数量与配置数量不一致，当前主题数量为{theme_num}，配置数量为{len(conf)}，"
                         f"已截断为 {min(theme_num, len(conf))} 处理")
        memory_fragments = []
        valid_count = min(theme_num, len(conf))
        for i in range(valid_count):
            # 负索引、零宽/反向区间已在 SummaryMemoryFragmentsConfig 的模型校验里兜底，
            # 这里只处理与消息条数相关的越界
            start = conf[i].start_idx
            end = min(conf[i].end_idx, len(messages))
            if start >= len(messages) or start >= end:
                logger.warning(f"第 {i} 个片段范围无效: [{start}, {end})，"
                               f"消息数 {len(messages)}，跳过该片段")
                continue

            content = "\n".join(text[start:end])
            # 取片段内最后一条消息的时间
            msg_end_time = messages[end - 1].additional_kwargs.get('time', time.time())
            metadata = MemoryFragmentsMetadata(
                theme=conf[i].theme,
                type=conf[i].type,
                time=msg_end_time,
                strengthen_num=0,
            )
            memory_fragments.append(MemoryFragments(
                config=metadata,
                content=content,
            ))
        return memory_fragments


class FragmentsMemoryRAG:
    def __init__(self):
        self.summary_llm = None
        self._init_lock = Lock()
        # sqlite 单连接不支持并发操作：中间件的锁是"按用户"的，跨用户/恢复消费者的写入
        # 会同时落到同一个连接上（InterfaceError）。此锁串行化所有 sql_vec 操作。
        # （并发回归测试见 memory_middleware 独立版 tests/test_user_isolation.py）
        self._sql_lock = asyncio.Lock()
        self.prompt = """
        
        你现在是一个专业的总结专家，你需要根据用户的新增对话片段来总结出相对应的用户画像。
        注意：
            - 只总结片段中明确出现的用户信息，不要猜测。
            - 片段中没有涉及的字段，保持默认值或空值即可（合并时会保留旧画像中的对应信息）。
        """
        if rag_db != 'sqlite':
            logger.error(f"rag数据库配置错误，仅支持sqlite，当前配置为{rag_db}")
            raise ValueError(f"rag数据库配置错误，仅支持sqlite，当前配置为{rag_db}")
        sql_url = rag_args['db_url']
        if not is_absolute_path(sql_url):
            sql_url = os.path.join(ROOT_BASE_DIR_PATH, sql_url)
        self.sql_vec = CustomizeSQLiteVec(
            table='memory_fragments',
            embedding=_embeddings,
            db_file=sql_url,
            connection=None,
            is_async=True,
        )
        self._con = sqlite3.connect(sql_url, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        # 并发优化：与 CustomizeSQLiteVec 的写连接共享同一数据库文件，
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA busy_timeout=5000")

    def max_fragment_ids(self, user_id: str) -> tuple:
        """该用户片段的高水位：(最大片段 id, 最大已切分消息数)。

        片段 id 计数器活在 Redis、片段本体在这里；Redis 一丢（重启 / 无持久化）
        计数器就从 1 重算并撞 UNIQUE 约束，故用库里的持久高水位兜底。
        msg_end 由写入端打在每次切分的最后一片 metadata 上；旧数据没这个键 → 取 0，
        行为与兜底前一致，无需迁移。
        """
        cursor = self._con.cursor()
        cursor.execute(
            "select max(cast(json_extract(metadata, '$.id') as integer)) as max_fid, "
            "max(cast(json_extract(metadata, '$.msg_end') as integer)) as max_offset "
            "from memory_fragments where json_extract(metadata, '$.user_id') = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        return (int(row['max_fid'] or 0), int(row['max_offset'] or 0))

    async def add(self, memory: Optional[List[Document]]) -> None:
        if not memory:
            logger.error("记忆片段不能为空")
            return None
        async with self._sql_lock:
            await self.sql_vec.aadd_documents(memory)

    async def query_context_distance(
            self,
            query: str,
            k: Optional[int] = None,
            user_id: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """检索与 query 最相似的记忆片段，按 user_id 过滤，避免跨用户泄露。

        k=None 表示不预筛（该用户全部片段都作为候选）：预筛按主题相似度取候选，主题一偏就会
        把载着细节的片段整块挡在门外。sqlite-vec 本就是全表算距离再取前 k，放开几乎零成本。
        """
        if k is None:
            k = 4096
        async with self._sql_lock:
            doc_scores = await self.sql_vec.asimilarity_search_with_score(
                query=query, k=k, filter={'user_id': user_id})
        if not doc_scores:
            return list()
        # sqlite-vec 返回的 score 为距离，cosine 距离 = 1 - 余弦相似度
        return [(doc, 1 - score) for doc, score in doc_scores]

    async def update_config_strengthen(self, ids: List[str], documents: List[Document]):
        """
        由于RAG不支持直接更新，所以采用先删除再添加的方式来达到更新的效果
        注意：
            该方法只用于更新记忆片段的强化次数，不用于更新记忆片段的内容或其他元数据
        """
        # 删+添在同一把锁内，避免与并发写入交错破坏数据
        async with self._sql_lock:
            await self.sql_vec.adelete(ids=ids)
            await self.sql_vec.aadd_documents(documents)

    async def long_memory_summary_by_time(
            self,
            model_name: str,
            m_t: float,
            m_c: float,
            long_term_value: float,
            user_id: str,
            last_summary_id: int = 0,
            **kwargs
    ) -> Optional[Tuple[dict, int]]:
        """
        对该用户在归纳游标（last_summary_id）之后、且最早片段已成熟的新片段做增量总结。
        返回：(增量画像 dict, 新的游标值)；无新片段或未成熟时返回 None。
        """
        await self.create_summary_agent(model_name, **kwargs)
        async with self._sql_lock:
            result = self.select_time_by_sqlite(m_t, m_c, long_term_value, user_id, last_summary_id)

        if not result:
            return None

        text, new_cursor = result
        # 此处添加失败保底机制
        try:
            profile = await self.summary_llm.ainvoke([
                SystemMessage(content=self.prompt),
                HumanMessage(content=f"目前的新增对话片段如下：\n{text}"),
            ])
        except Exception as e:
            logger.error(f"在进行总结分片时，出现异常：{e}")
            # 将未处理的text放入RabbitMQ中，后续恢复后重试
            try:
                exchange = await _ensure_rabbit_ready()
                await exchange.publish(
                    Message(
                        body=ErrorFragmentsData(
                            type='summary',
                            user_id=user_id,
                            text=text,
                            last_summary_id=new_cursor
                        ).model_dump_json().encode(),
                        delivery_mode=DeliveryMode.PERSISTENT,
                        content_type='application/json',
                        timestamp=time.time(),
                    ),
                    routing_key=summary_routing_key,
                    mandatory=False,
                )
            except Exception as publish_e:
                logger.error(f"发送错误消息到RabbitMQ失败：{publish_e}")
            finally:
                email_result = send_error_email(subject="RAG总结分片异常", content=f"在进行总结分片时，出现异常：{e}")
                if not email_result:
                    logger.error("发送错误邮件失败, 请查阅相关的日志信息来锁定问题")
            # 返回None，而不选择保底机制，是因为UserProfile类过于复杂，且关乎到用户的隐私和安全
            profile = None
        if not profile:
            return None
        profile_json = profile.model_dump(mode='json')
        profile_json['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"用户画像（增量总结）：{profile_json}")
        return profile_json, new_cursor

    def select_time_by_sqlite(self, m_t: float, m_c: float, long_term_value: float,
                              user_id: str, last_summary_id: int) -> Optional[Tuple[str, int]]:
        cursor = self._con.cursor()
        # 仅取该用户在归纳游标之后的新片段；fid 为 metadata 中的数值型片段序号
        cursor.execute(
            "select text, metadata, cast(json_extract(metadata, '$.id') as integer) as fid "
            "from memory_fragments "
            "where json_extract(metadata, '$.user_id') = ? "
            "and cast(json_extract(metadata, '$.id') as integer) > ? "
            "order by fid",
            (user_id, last_summary_id),
        )
        row = cursor.fetchall()
        cursor.close()
        if not row:
            return None

        metadata = json.loads(row[0]['metadata'])
        memory_time = metadata.get('time', time.time())
        m = 1 - math.exp(-((time.time() - memory_time) / m_t) ** m_c)
        if m < long_term_value:
            return None
        text_list = [item['text'] for item in row]
        return '\n'.join(text_list), row[-1]['fid']

    async def create_summary_agent(self, model_name: str, **kwargs):
        if self.summary_llm is None:
            async with self._init_lock:
                if self.summary_llm is None:
                    _init_llm = init_chat_model(model_name, **kwargs)
                    self.summary_llm = _init_llm.with_structured_output(UserProfile).with_retry(
                        retry_if_exception_type=(Exception,),
                        wait_exponential_jitter=True,
                        stop_after_attempt=3,
                        exponential_jitter_params=ExponentialJitterParams(
                            initial=2.0,
                            max=10,
                            exp_base=2,
                            jitter=1.0
                        ),
                    )
