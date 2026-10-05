from datetime import datetime
from typing import Literal, Optional, List

from pydantic import BaseModel, Field, model_validator

from SPO.user import DeskSalespersonUserPersona

# identity: 包含用户身份信息（姓名、职业等）
# preference: 包含用户偏好、习惯
# decision: 包含关键决策、承诺
# fact: 包含客观事实、技术细节
# episode: 描述某个经历或任务过程
# chat: 一般性闲聊，长期价值低
fragments_type = Literal["identity", "preference", "decision", "fact", "episode", "chat"]


class UserProfile(BaseModel):
    """
    结构化用户信息模型
    用于大模型推理用户画像，进行跨会话记忆
    """
    # --- 核心身份 ---
    user_name: str = Field(default='', description="用户名")
    user_sex: Literal['男', '女', '保密'] = Field(default='保密', description="用户性别")
    user_age: Optional[int] = Field(default=None, description="用户年龄")
    user_occupation: Optional[str] = Field(default=None, description="职业")

    # --- 偏好与风格 ---
    user_hobby: list[str] = Field(default=[],
                                  description="用户爱好列表(爱好：在兴趣基础上经过时间和实践形成的持久喜好.)")
    user_interest: list[str] = Field(default=[], description="用户兴趣列表(兴趣：是对某事物的短期关注或好奇.)")
    user_speak_style: str = Field(default='', description="用户说话风格")
    user_communication_style: Optional[
                                  Literal['直截了当', '委婉', '幽默', '正式']
                              ] | str = Field(default='', description="用户沟通风格")
    # --- 知识与能力 ---
    user_expertise: list[str] = Field(default_factory=list, description="专业领域")

    # --- 当前上下文 ---
    user_goals: list[str] = Field(default_factory=list, description="当前目标/正在进行的项目")
    user_constraints: list[str] = Field(default_factory=list, description="明确约束条件")
    user_values: list[str] = Field(default_factory=list, description="核心价值观")

    # --- 关系网络 ---
    key_relationships: dict[str, str] = Field(default_factory=dict, description="重要关系人与角色")

    # --- 业务相关 ---
    user_person: Optional[DeskSalespersonUserPersona] = Field(
        default=None,
        description="用户业务画像(可选，默认为None), 可选的字段为："
                    "preference-用户喜好, hate-用户的憎恶点, dislike_goods_id-用户不喜欢的商品id列表, "
                    "dislike_brand-用户不喜欢的品牌列表, dislike_style-用户不喜欢的风格, like_goods_id-用户喜欢的商品id列表, "
                    "like_brand-用户喜欢的品牌列表, like_style-用户喜欢的风格"
    )

    # --- 元数据 ---
    last_updated: Optional[datetime] = Field(default=None,
                                             description="最后更新时间, 系统会自动补充，无需添加，但要一起返回")


class MemoryFragmentsMetadata(BaseModel):
    theme: str = Field(default='', description="记忆主题")
    type: List[fragments_type] = Field(default=[], description="在同一主题下可能存在的类型")
    time: float = Field(default=0.0, description="记忆时间")
    strengthen_num: int = Field(default=0, description="巩固次数")


class MemoryFragments(BaseModel):
    """
    用于存储用户记忆片段的模型
    """
    config: MemoryFragmentsMetadata = Field(default_factory=MemoryFragmentsMetadata)
    content: str = Field(default='', description="记忆内容")


class SummaryMemoryFragmentsConfig(BaseModel):
    theme: str = Field(default='', description="记忆主题")
    type: List[fragments_type] = Field(default=[], description="在同一主题下可能存在的类型")
    start_idx: int = Field(default=0, description="片段起始消息索引（从0开始计数，左闭）")
    end_idx: int = Field(default=1, description="片段结束消息索引（左开：实际覆盖到 end_idx-1 那条消息）")

    @model_validator(mode='after')
    def _normalize_message_range(self):
        """结构化输出注入后即兜底：负索引归零；零宽/反向区间退化为只取 start_idx 那一条。"""
        if self.start_idx < 0:
            self.start_idx = 0
        if self.end_idx <= self.start_idx:
            self.end_idx = self.start_idx + 1
        return self


class SummaryMemoryAi(BaseModel):
    theme_num: int = Field(default=0, description="长对话的主题总数，必须与config列表长度一致")
    config: List[SummaryMemoryFragmentsConfig] = Field(
        default_factory=list,
        description="记忆片段配置列表，每个配置对应一个主题的摘要规则"
    )
    current_theme: str = Field(default='', description="当前正在处理的主题")
