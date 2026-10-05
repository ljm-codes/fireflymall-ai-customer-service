import math
import time
import warnings
from asyncio import Lock
from typing import Any, Optional, Callable, Awaitable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.summarization import TokenCounter
from langchain.agents.middleware.types import ResponseT, ExtendedModelResponse, ToolCallRequest
# from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, HumanMessage
# from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from typing_extensions import override
from langchain.chat_models import init_chat_model

from SPO.memory import UserProfile
from SPO.state import MainState
from Tools.log_settings import LogSetting
from .memory_rag import MemoryFragmentsAiSpliter, FragmentsMemoryRAG
from .prompt import SystemPromptOperation
from .token_calculate import TokenCalculatorFactory
from load_config.config import config

logger = LogSetting.create(logger_name=__name__)

_TYPE_SCORE_MAP = {
    # 自我相关记忆具有 自我参照效应（Rogers et al., 1977），回忆率最高
    'identity': 0.95,
    # 决策记忆涉及脚本记忆（Schank & Abelson），高重复调用性
    'decision': 0.85,
    # 偏好属于内隐态度，稳定性高但检索频率中等
    'preference': 0.80,
    # 语义记忆，稳定但情感附着低，易受干扰
    'fact': 0.60,
    # 情景记忆，易受时间衰减影响最大
    'episode': 0.40,
    # 无结构闲聊，遗忘曲线最陡
    'chat': 0.15
}

_VOCATION_PARAM_MAP = {
    'collaborative creation': {
        'M(Δt)': {
            'τ_m': 3600,
            'c': 1.0
        },
        'T(m)': {
            'τ': 604800,
            'c': 0.7
        },
        'slice': 0.9,
        'long-term': 0.98
    },
    'customer service': {
        'M(Δt)': {
            'τ_m': 600,
            'c': 0.5
        },
        'T(m)': {
            'τ': 43200,
            'c': 0.5
        },
        'slice': 0.7,
        'long-term': 0.9
    },
    'accompany': {
        'M(Δt)': {
            'τ_m': 7200,
            'c': 0.8
        },
        'T(m)': {
            'τ': 86400,
            'c': 0.5
        },
        'slice': 0.8,
        'long-term': 0.95,
    }
}

_METERAGE_MAP = {
    'K': 10 ** 3,
    'M': 10 ** 6,
    'B': 10 ** 9,
}


class TimeMemoryFormulaParam(BaseModel):
    w0: float = Field(default=0.1, description='基础重要性')
    w1: float = Field(default=0.8, description='类型重要性的权重')
    w2: float = Field(default=0.1, description='巩固次数的权重')
    alpha: float = Field(default=0.33, description='语义相关性权重')
    beta: float = Field(default=0.33, description='时间衰减权重')
    gamma: float = Field(default=0.34, description='固有重要性权重')
    delta: float = Field(default=0.0, description='基线保留偏移, 独立于归一化约束，仅用于微调')


def clamp(value, min_val, max_val):
    if min_val >= max_val:
        raise ValueError(f'不允许出现min_val >= max_val的情况')
    return max(min_val, min(value, max_val))


async def _load_user_profile(user_id: str, runtime: Runtime) -> dict:
    """读取用户既有的长期记忆画像（含归纳游标），失败时按空画像处理"""
    if runtime.store is None:
        return {}
    try:
        item = await runtime.store.aget(('long-term', 'user_profile',), user_id)
    except Exception as e:
        logger.warning(f"读取用户画像失败，按空画像处理: {e}")
        return {}
    if item is None:
        return {}
    return item.value.get('user_profile', {}) or {}


class BalancedMultiDimensionMemory(AgentMiddleware):
    """
    衡忆多维认知架构 (BalancedMultiDimensionMemory)
    结合 AI 模型，与相对应的数学公式，解决：
    多维加权评分 + 可调艾宾浩斯衰减 + 主题分片 + 权重制衡 + 分层记忆 + 防上下文分裂

    架构讲解：
        一些概念：
            1、工作记忆（短期记忆）：带有时间戳的messages列表
            2、记忆暂存库（待巩固的中间层）：从工作记忆卸载下来的原始对话切片，是长期记忆的原材料。这还不是真正的长期记忆。
                存储方式：
                    向量数据库，存储带元数据的片段：
                        - 情境标签（主题、参与角色）
                        - 时间信息（t_created）
                        - 原始对话或摘要
            3、长期记忆库（经巩固的知识层）：从暂存库中提炼和抽象出的结构化知识，具有高持久性和高重要性。
                存储字段：1、事实 2、偏好 3、方法 4、情境 ...

        流程：
        1、第一步(输入与暂存)：
            (1)、首先先判断消息是否有时间戳，若没有则添加时间属性。之后再系统周期性的参加对话的“记忆成熟度M”
                M(Δt)=1-exp(-(Δt/τ_m)^c) τ_m为控制固化速度。
                当最早消息的M值超过某个阈值（如0.8）时，进行切片
            (2)、切片时，调用轻量的LLM为该切片生成情境标签（主题等）,将这些除内容以外的元数据（MemoryFragmentsMetadata）也存储起来。
            (3)、将切片好的片段存放到记忆暂存库（待巩固的中间层），但此时不要将原本的消息从工作记忆中移除。
                当达到['fraction', 'tokens', 'messages'] 中的一个阈值后，进行裁取。
        2、第二步(检索与巩固)：
            (1)、当进入裁取时，需要根据当前任务/事件，来动态调整α, β, γ 的参数值，
                α, β, γ 具有约束：
                    范围约束：
                        α   语义相关的权重      [0, 1]，默认值 0.33
                        β   时间衰减的权重      [0, 1]，默认值 0.33
                        γ   固有重要性的权重    [0, 1]，默认值 0.34
                        δ   基线保留偏移       [-0.1, 0.2]，默认值 0.0
                    逻辑约束：
                        要满足 α+β+γ=1.0
                        if α * R > 0.7 * S，则暂时将 α 下调至 0.7 * S / R
                        根据当前任务/事件,让LLM来决定α,β,γ,δ的取值。
            (2)、计算 S(m)=α*R(m,q)+β*T(m)+γ*F(m)+δ
                S(m): 记忆片段得分
                R(m, q) ：语义相关性 -> 决定是否需要
                 - 具体公式：max(0, cos(m, q)-θ_min)
                    m: 记忆片段
                    q: 当前问题
                    θ_min: 最小相似度阈值(硬阈值)
                    cos(m, q): 余弦相似度，范围 [-1, 1]
                T(m) ：结合艾宾浩斯曲线的参数化时间衰减 -> 决定是否“过期”
                 - 具体公式：exp(-(Δt/τ)^c)
                    Δt = 当前时间 - 记忆的“最后一次强化时间”,
                    τ: 时间尺度参数，控制多久之后记忆强度降至原来的约 37%（即 1/e）。τ 越大，遗忘越慢。
                    c: 曲线形状参数
                        -> c = 1时，标准指数衰减（类似放射性衰变）
                        -> c < 1时，比指数衰减更先快后慢，非常像艾宾浩斯的早期形状。
                        -> c > 1时，开始时遗忘较慢，然后加速，形成“突然遗忘”的效应。
                F(m) ：固有重要性 -> 决定是否重要 -> 理应来说不用重复计算，可以直接在记忆暂存库中获取
                    F(m) = clamp(w0 + w1*∑(v_i*I_type(i))+w2*(1-exp(-refresh_count(m)*k)), 0, 1)
                    w0: 基础重要性(避免出现零分)
                    w1, w2: 限制权重
                    I_type(m): 类型重要性，由记忆的元数据决定。如：用户姓名(0.9)，临时聊天内容(0.2)
                    refresh_count(m): 再巩固次数，即这个记忆片段被成功检索并用于生成正确回答的次数
                    clamp 到 [0, 1]，防止权重溢出
            (3)、根据 S(m) 对记忆(m) 进行排序，取top_K。
            若这top_K被成功注入并辅助生成有效回复后，该片段发生再巩固：
                1、刷新其时间戳，影响后续的 T 值。
                2、适当增加其 F 值。
        3、第三步(记忆整理 -> 将老旧记忆归纳总结为长期记忆)：
            (1)、延续M值公式，再结合积累量来触发是否归纳总结。
                即：(M 值达到阈值) AND (积累量达到阈值) → 触发对该主题片段的记忆整理
                    记忆整理将会总结为一个结构化的记忆，存储到长期记忆库中。

    注：自定义配置信息放于 config.yaml 中
    """

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        super().__init__()
        self.vocation = config['model']['vocation']
        self.conf = config['model']['summary']
        self.chat_conf = config['model']

        self.model = self.conf['name']  # 模型名称
        self.chat_model = self.chat_conf['name']  # 聊天模型名称

        self.kwargs = self.conf['kwargs']  # 模型参数
        self.pattern = self.conf['pattern']  # 选择总结模式，可选值为 'fraction' 'tokens' 'messages'
        self.trigger = self.conf['trigger_threshold']  # 触发总结阈值

        if self.vocation['name'] not in ('collaborative creation', 'customer service', 'accompany', 'customize'):
            raise ValueError(
                "职业必须为 collaborative creation, customer service, accompany中的一个, 或者选择 customize")

        if self.vocation['name'] == 'customize' and not self.vocation.get('kwargs', None):
            # 自定义模式选择 自定义调参
            logger.warn(
                "自定义模式下，必须在 config.yaml 中配置 model.vocation.kwargs，默认采用 collaborative creation 模式参数")
            self.vocation['name'] = 'collaborative creation'

        if self.vocation['name'] == 'customize':
            self.vocation_kwargs = self.vocation['kwargs']
        else:
            self.vocation_kwargs = _VOCATION_PARAM_MAP.get(self.vocation['name'], {})

        if not self.vocation_kwargs:
            raise ValueError("模型参数为空，请检查代码或者 config.yaml 中是否配置了 model.vocation.kwargs")

        self.m_t = self.vocation_kwargs['M(Δt)']['τ_m']
        self.m_c = self.vocation_kwargs['M(Δt)']['c']
        self.t_t = self.vocation_kwargs['T(m)']['τ']
        self.t_c = self.vocation_kwargs['T(m)']['c']
        self.slice_value = self.vocation_kwargs['slice']
        self.long_term_value = self.vocation_kwargs['long-term']

        hf_id = self.chat_conf['huggingface_id'] if 'huggingface_id' in self.chat_conf else None

        self.token_calculator = TokenCalculatorFactory(token_counter, self.chat_model, hf_id)

        if self.pattern not in ['fraction', 'tokens', 'messages']:
            raise ValueError("总结模式必须为 fraction, tokens, messages 中的一个")

        self.max_token = self._get_model_max_tokens()
        self.kwargs.get('profile')['max_input_tokens'] = self.max_token
        if self.pattern == 'fraction':
            if self.max_token is None:
                raise ValueError(
                    "模型未正确且合理配置最大token数，必须要在 config.yaml 中配置 model.summary.kwargs.profile.max_input_tokens")
            if not 0 <= self.trigger <= 1:
                raise ValueError("对于 fraction 模式，总结阈值必须在 0 到 1 之间")
            if self.trigger < 0.7:
                warnings.warn("对于 fraction 模式，总结阈值过低，建议设置为 0.7 或以上，以避免反复总结")
        elif self.pattern == 'tokens':
            if not isinstance(self.trigger, int):
                raise ValueError("对于 tokens 模式，总结阈值必须为整数")
            if self.trigger <= 0:
                raise ValueError("对于 tokens 模式，总结阈值必须大于 0")
            if self.trigger < 10000:
                warnings.warn("对于 tokens 模式，总结阈值过低，建议设置为 10000 或以上，以避免反复总结")
        elif self.pattern == 'messages':
            if not isinstance(self.trigger, int):
                raise ValueError("对于 messages 模式，总结阈值必须为整数")
            if self.trigger <= 0:
                raise ValueError("对于 messages 模式，总结阈值必须大于 0")
            if self.trigger < 50:
                warnings.warn("对于 messages 模式，总结阈值过低，建议设置为 50 或以上，以避免反复总结")

        self._summary_llm = None
        self.memory_spliter = MemoryFragmentsAiSpliter(self.model, self.kwargs)
        self.memory_fragments_rag = FragmentsMemoryRAG()
        # 片段 id 撞库兜底：Redis 计数器丢失（重启 / 无持久化）后从向量库持久高水位接着编
        self.memory_spliter.id_source = self.memory_fragments_rag
        from agent.main_agent import main_system_prompt
        self._main_system_prompt = main_system_prompt
        # 按 user 隔离的提示词操作实例，避免多用户并发时互相覆盖 memory_fragments
        self._prompt_operations: dict[str, SystemPromptOperation] = dict()
        # 调参缓存：(user_id, 主题) → 参数。主题只在切片时变，同一主题下的多次注入不必重复调 LLM
        self._param_cache: dict = dict()
        self.math_agent_prompt = """
        你现在是一个专业的数学专家，你需要通过当前聊天主题来调整对于公式的参数。
        公式：S(m)=α*R(m,q)+β*T(m)+γ*F(m)+δ
        其中：
            S(m) -> 记忆片段得分
            R(m, q) ：语义相关性 -> 决定是否需要
            T(m) ：结合艾宾浩斯曲线的参数化时间衰减 -> 决定是否“过期”
            F(m) ：固有重要性 -> 决定是否重要
            而F(m) 的公式为：clamp(w0 + w1*∑(v_i*I_type(i))+w2*(1-exp(-refresh_count(m)*k)), 0, 1)
                refresh_count(m) ：记忆m的刷新次数，初始值为0，每次刷新增加1
                w0, w1, w2：超参数，根据实际情况调整
            α, β, γ, δ ：超参数，根据实际情况调整
        
        因此，你需要根据当前聊天主题，调整w0, w1, w2, α, β, γ, δ的值。
        
        参数限制：
            参数 建议范围         说明
            w0  [0.05, 0.15]    基础重要性，确保任何记忆片段都不会因零分而被完全遗忘。值不宜过大，否则会稀释类型和巩固的区分度。
            w1  [0.5, 0.8]      类型重要性的权重，是 F(m) 的主要贡献项。因为类型是记忆最稳定的属性，决定其先天重要性。
            w2  [0.05, 0.2]     巩固次数的权重，是记忆的后天强化项。值不宜过大，否则频繁访问的片段会过度压制重要但很少被检索的关键信息。每次巩固的增量很小（如每次 +0.02），需要通过多次巩固才能显著提升。
            
            α   [0, 1]          语义相关性权重
            β   [0, 1]          时间衰减权重
            γ   [0, 1]          固有重要性权重
            δ   [-0.1, 0.2]     基线保留偏移, 独立于归一化约束，仅用于微调
        
        必须满足：
            w0+w1+w2=1
            α+β+γ=1
        """

        self._lock: dict[str, Lock] = dict()
        self._user_retrieve_state: dict[str, bool] = dict()
        self._init_lock = Lock()

    async def _ensure_summary_agent(self):
        if self._summary_llm is None:
            async with self._init_lock:
                if self._summary_llm is None:
                    initial_model = init_chat_model(self.model, **self.kwargs)
                    self._summary_llm = initial_model.with_structured_output(TimeMemoryFormulaParam)

    @override
    async def abefore_model(self, state: MainState, runtime: Runtime) -> dict[str, Any] | None:
        await self._ensure_summary_agent()
        new_idx = state.get('new_msg_idx', 0)  # type: ignore
        messages = state['messages'][new_idx:]

        # 自包含时间戳兜底：正常情况下用户消息由 msg_handle 入口打点，
        # 但异常路径/外部未打点时在此补齐（缺失视为当前时间，幂等不覆盖）。
        now = time.time()
        for msg in messages:
            if 'time' not in msg.additional_kwargs:
                msg.additional_kwargs['time'] = now
                logger.debug(f"消息缺失 time 字段，已按首次见到时间补点: {msg.content[:50]!r}")

        user_id = runtime.context.user_id
        self._lock.setdefault(user_id, Lock())

        async with self._lock[user_id]:
            """
            记忆切片
            """
            self._user_retrieve_state.setdefault(user_id, False)

            if not await self._memory_slice(messages, model_name=self.model, user_id=user_id, runtime=runtime,
                                            **self.kwargs):
                return None

            current_tokens = self._calculate_current_token(messages, user_id)

            if self._is_primary_triggered(current_tokens, len(messages)):
                # 事件 = 窗口达到预算。awrap_model_call 会把上下文截断到最近一条人类消息，
                # 因此固化必须与事件同刻发生，否则这段内容既不在上下文、也不在暂存库。
                # 切分不再是独立动作：一次事件 = 切分 + 调参 +（片段成熟时）归纳。
                # text 必须传完整消息列表（消息偏移游标在 spliter 内部管理，见 atext_to_document 说明）
                if memory_fragments := await self.memory_spliter.atext_to_document(text=state['messages'], user=user_id):
                    try:
                        await self.memory_fragments_rag.add(memory_fragments)
                    except Exception as e:
                        # 落库失败不该掀掉整轮对话：游标已推进，本批片段弃掉（留 id 空洞但不会重复）
                        logger.error(f"记忆片段落库失败，跳过本批: {e}")
                self._user_retrieve_state[user_id] = True
                return None

            return None

    @override
    async def awrap_model_call(
            self,
            request: ModelRequest[ContextT],
            handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        user_id = request.runtime.context.user_id
        system_prompt = request.state.get('system_prompt')  # type: ignore
        msg_idx = request.state.get('new_msg_idx')  # type: ignore
        logger.info(f"用户ID：{user_id}")
        try:
            if not self._user_retrieve_state.get(user_id, False):
                if system_prompt is not None and msg_idx is not None:
                    new_request = request.override(
                        system_message=SystemMessage(content=system_prompt),
                        messages=request.messages[msg_idx:],
                    )
                    return await handler(new_request)
                else:
                    return await handler(request)
            # 提取返回、重写
            related_fragments = await self._extract_fragments_by_time(user_id)
            if not related_fragments:
                # 当前无可注入的记忆片段：重置检索状态，按普通调用处理
                self._user_retrieve_state[user_id] = False
                if system_prompt is not None and msg_idx is not None:
                    new_request = request.override(
                        system_message=SystemMessage(content=system_prompt),
                        messages=request.messages[msg_idx:],
                    )
                    return await handler(new_request)
                else:
                    return await handler(request)

            # 按 user 隔离提示词操作实例，避免多用户并发时互相覆盖
            if user_id not in self._prompt_operations:
                self._prompt_operations[user_id] = SystemPromptOperation(initial_prompt=self._main_system_prompt)
            prompt_ops = self._prompt_operations[user_id]
            new_documents, old_ids = prompt_ops.add_memorys([x[0] for x in related_fragments])
            # 用户画像作为独立、每轮新鲜的层
            prompt_ops.prompt.profile = request.state.get('user_profile') or ''
            new_prompt = prompt_ops.get_prompt()
            logger.info(f"重写后的提示词为：{new_prompt}")

            total_messages = len(request.messages)
            for msg in reversed(request.messages):
                total_messages -= 1
                if isinstance(msg, HumanMessage):
                    break
            if total_messages < 0:
                total_messages = len(request.messages) - 1

            new_request = request.override(
                system_message=SystemMessage(content=new_prompt),
                messages=request.messages[total_messages:],
            )
            self._user_retrieve_state[user_id] = False

            response = await handler(new_request)
            # 模型调用成功后才将强化落库，避免失败轮次误记巩固；
            # 强化失败属于尽力而为的后置操作，不应触发外层降级导致模型被二次调用
            try:
                await self.memory_fragments_rag.update_config_strengthen(ids=old_ids, documents=new_documents)
            except Exception as e:
                logger.error(f"记忆强化落库失败（不影响本次回复）: {e}")
            request.state['new_msg_idx'] = total_messages  # type: ignore
            state_update = {"new_msg_idx": total_messages, "system_prompt": new_prompt}
            return ExtendedModelResponse(model_response=response, command=Command(update=state_update))
        except Exception as e:
            self._user_retrieve_state[user_id] = False
            logger.error(f"记忆检索失败，降级为普通调用: {e}")
            return await handler(request)

    @override
    async def aafter_model(
            self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """
        为 AI 消息添加时间戳
        """
        messages = state['messages']
        messages = list(reversed(messages))
        for msg in messages:
            if isinstance(msg, AIMessage) and 'time' not in msg.additional_kwargs:
                msg.additional_kwargs['time'] = time.time()
                break

        return None

    @override
    async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """
        在工具执行完成的当刻为 ToolMessage 打上真实时间戳，
        避免在下一轮的 abefore_model 中补打 time.time() 造成时间偏差
        """
        result = await handler(request)
        if isinstance(result, ToolMessage) and 'time' not in result.additional_kwargs:
            result.additional_kwargs['time'] = time.time()
        return result

    def _get_model_max_tokens(self):
        if 'profile' not in self.kwargs:
            return None
        profile = self.kwargs['profile']
        if 'max_input_tokens' not in profile:
            return None

        max_tokens = profile['max_input_tokens']
        # 若是数字直接返回
        if isinstance(max_tokens, int):
            return max_tokens
        try:
            # 若是字符串，则进行转换
            last_symbol = max_tokens[-1].upper()
            num_tokens = int(max_tokens[:-1])
            if last_symbol in _METERAGE_MAP:
                return num_tokens * _METERAGE_MAP[last_symbol]
            return None
        except ValueError as e:
            raise ValueError(f"max_input_tokens 无效: {e}")

    def _calculate_current_token(self, messages: list[BaseMessage], user_id: str = None) -> int:
        if user_id is None:
            raise ValueError("user_id cannot be None")
        count_token = self.token_calculator.calculate_tokens(messages)
        return math.ceil(count_token)

    def _is_primary_triggered(self, segment_tokens: int, segment_len: int) -> bool:
        """
        判断是否触发一次总结
        参数：
            segment_tokens: 段落token数
            segment_len: 段落长度
        返回：
            是否触发一次总结
        """
        if self.pattern == 'fraction':
            return segment_tokens >= self.max_token * self.trigger
        elif self.pattern == 'tokens':
            return segment_tokens >= self.trigger
        elif self.pattern == 'messages':
            return segment_len >= self.trigger

    @staticmethod
    async def _summary_to_long_term(res_json: dict, user_id: str, runtime: Runtime):
        # 总结放在长期记忆中，文件放在RAG数据库中
        if runtime.store is None:
            logger.warning("未配置 langgraph store，跳过长期记忆存储")
            return
        if res_json.get('user_person'):
            await runtime.store.aput(
                ('long-term', 'user_person',),
                user_id,
                {'user_person': res_json.get('user_person')}
            )
        await runtime.store.aput(
            ('long-term', 'user_profile',),
            user_id,
            {'user_profile': res_json}
        )

    @staticmethod
    def _field_default_value(field_name: str):
        """获取 UserProfile 字段的默认值，用于判断增量总结中哪些字段未被提及"""
        fld = UserProfile.model_fields.get(field_name)
        if fld is None:
            return None
        if fld.default is not PydanticUndefined:
            return fld.default
        if fld.default_factory is not None:
            return fld.default_factory()
        return None

    @classmethod
    def merge_user_profile(cls, old: dict, new: dict) -> dict:
        """
        将增量总结的新画像合并进旧画像：
        - 列表字段：去重合并（新值优先）
        - 字典字段：递归合并
        - 标量字段：仅当新值非空且不是模型默认值时才覆盖，避免增量总结把未提及的字段重置为默认值
        """
        merged = dict(old)
        for key, new_val in new.items():
            old_val = merged.get(key)
            if isinstance(new_val, list):
                combined = list(new_val)
                if isinstance(old_val, list):
                    for item in old_val:
                        if item not in combined:
                            combined.append(item)
                merged[key] = combined
            elif isinstance(new_val, dict):
                merged[key] = {**(old_val if isinstance(old_val, dict) else {}), **new_val}
            else:
                default = cls._field_default_value(key)
                if new_val not in (None, '', default):
                    merged[key] = new_val
        return merged

    async def _memory_slice(
            self,
            messages: list[BaseMessage],
            model_name: str,
            user_id: str,
            runtime: Runtime,
            **kwargs
    ):
        """
        是否进行记忆切片
        """
        last_msg = messages[0]
        last_time = last_msg.additional_kwargs.get('time', time.time())
        m = 1 - math.exp(-((time.time() - last_time) / self.m_t) ** self.m_c)

        # 增量归纳：读取既有画像的游标，只对游标之后的新片段做总结，并合并进旧画像
        existing_profile = await _load_user_profile(user_id, runtime)
        last_summary_id = existing_profile.get('last_summarized_id', 0)
        if profile := await self.memory_fragments_rag.long_memory_summary_by_time(
                model_name, self.m_t, self.m_c, self.long_term_value,
                user_id=user_id, last_summary_id=last_summary_id, **kwargs):
            profile_json, new_cursor = profile
            merged = self.merge_user_profile(existing_profile, profile_json)
            merged['last_summarized_id'] = new_cursor
            await self._summary_to_long_term(merged, user_id, runtime)
        logger.info(f"记忆切片概率：{m}")
        return m >= self.slice_value

    async def _extract_fragments_by_time(self, user_id: str):
        current_theme_by_user = self.memory_spliter.dialogue_theme_by_user.get(user_id, '')
        if not current_theme_by_user:
            return None
        # list[tuple[Document, float]]
        # TODO: 未来将提升此函数的对于异常的处理，并解决旧文档的问题
        #   当前问题：若memory_rag.py中的spliterLLM失效，那么数据库历史消息将不会更新，那么到时候永远只会提取同个片段
        #       导致出现LLM选取无效片段，内容不准确。
        #   由于当前所有进行分片、总结等模型就会同一个，那么若一个失效，这将会整体失效。因此，本次提交将不改。
        all_fragments_by_theme = await self.memory_fragments_rag.query_context_distance(
            current_theme_by_user, k=None, user_id=user_id)
        if not all_fragments_by_theme:
            # 库里还没有该用户的片段时，跳过 LLM 调参调用，避免每轮白烧一次模型
            return None
        cache_key = (user_id, current_theme_by_user)
        params_class = self._param_cache.get(cache_key)
        if params_class is None:
            # 同一 (user, 主题) 不重复调参：主题由切分模型产出、只在切片时变，绝大多数注入都能命中缓存
            params_class = await self._summary_llm.ainvoke([
                SystemMessage(content=self.math_agent_prompt),
                HumanMessage(content=f"当前用户{user_id}的对话主题：{current_theme_by_user}"),
            ])
            if len(self._param_cache) >= 256:   # FIFO 淘汰，防无界增长
                self._param_cache.pop(next(iter(self._param_cache)))
            self._param_cache[cache_key] = params_class
        w0 = params_class.w0
        w1 = params_class.w1
        w2 = params_class.w2
        alpha = params_class.alpha
        beta = params_class.beta
        gamma = params_class.gamma
        delta = params_class.delta

        retrieve_fragments = []
        for document, cosine in all_fragments_by_theme:
            document_time = document.metadata['time']
            type_list = document.metadata['type']
            strengthen_num = document.metadata['strengthen_num']
            # cosine取值范围为 [-1, 1]，取0.5 是因为cosine 0.5以上的相关度较高，更符合对应的主题
            relevance = cosine - 0.5
            r = relevance if relevance > 0 else 0
            t = math.exp(-((time.time() - document_time) / self.t_t) ** self.t_c)
            # clamp(w0 + w1*∑(v_i*I_type(i))+w2*(1-exp(-refresh_count(m)*k)), 0, 1)
            f = clamp(w0 + w1 * sum(_TYPE_SCORE_MAP.get(t, 0.0) * 1.0 for t in type_list) + w2 * (
                    1 - math.exp(-strengthen_num * 0.5)), 0, 1)
            s_m = alpha * r + beta * t + gamma * f + delta
            retrieve_fragments.append((document, s_m))
        # 保底：identity / preference 类片段优先占席（与 _TYPE_SCORE_MAP 的最高两档 0.95/0.80 一致）——
        # 候选池里主题一偏这两类会整块落选，而它们正是公式自己声明的最重要记忆
        _priority = {'identity', 'preference'}
        retrieve_fragments.sort(
            key=lambda x: (not (_priority & set(x[0].metadata.get('type') or [])), -x[1]))
        return retrieve_fragments[:4]
