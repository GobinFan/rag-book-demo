import os
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 定义计费相关的路由及其对应的响应动作
def billing_response(query):
    # 这里可以定义生成响应的逻辑，例如使用RAG模型
    return "感谢您的询问，我们将帮助您处理计费问题。"

billing = Route(
    name="Billing",
    utterances=[
        "我想取消订阅",
        "我想升级我的服务",
        "如何添加付款方式",
        "为什么我的账户被收费",
        "我想要争议这笔费用",
    ],
    function_call=billing_response,  # 将动作与路由关联
)

# 定义技术支持相关的路由及其对应的响应动作
def tech_support_response(query):
    # 这里可以定义生成响应的逻辑，例如使用RAG模型
    return "技术支持团队随时为您服务，请告诉我们您的问题。"

tech_support = Route(
    name="Technical Support",
    utterances=[
        "我的设备出问题了",
        "这个软件兼容我的设备吗",
        "我需要更新我的软件",
    ],
    function_call=tech_support_response,  # 将动作与路由关联
)

# 定义账户管理相关的路由及其对应的响应动作
def account_management_response(query):
    # 这里可以定义生成响应的逻辑，例如使用RAG模型
    return "账户管理团队将协助您处理账户相关的问题。"

account_management = Route(
    name="Account Management",
    utterances=[
        "我想更改我的密码",
        "我忘记了用户名",
        "如何更新我的个人信息",
    ],
    function_call=account_management_response,  # 将动作与路由关联
)

# 定义一般咨询相关的路由及其对应的响应动作
def general_inquiry_response(query):
    # 这里可以定义生成响应的逻辑，例如使用RAG模型
    return "欢迎咨询，我们将尽力为您提供帮助。"

general_inquiry = Route(
    name="General Inquiry",
    utterances=[
        "你们的服务有哪些",
        "我如何联系客服",
        "你们的公司地址在哪里",
    ],
    function_call=general_inquiry_response,  # 将动作与路由关联
)

# 将所有可能的 route 放在一起
routes = [billing, tech_support, account_management, general_inquiry]

# 用于对 query 进行 encode
encoder = OpenAIEncoder()

# 创建路由层
rl = RouteLayer(encoder=encoder, routes=routes)

# 测试路由选择和响应生成
print(rl("我想取消订阅"))  # 应该调用billing_response
print(rl("我的设备出问题了"))  # 应该调用tech_support_response
print(rl("我想更改我的密码"))  # 应该调用account_management_response
print(rl("你们的服务有哪些"))  # 应该调用general_inquiry_response

# 多路由检索，带相关性分数
# 输出多个路由选择及其相似性分数
print(rl.retrieve_multiple_routes("我需要更新我的软件，但是不知道如何操作"))
