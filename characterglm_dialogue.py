"""
一个简单的demo，调用CharacterGLM实现角色扮演，调用CogView生成图片，调用ChatGLM生成CogView所需的prompt。

依赖：
pyjwt
requests
streamlit
zhipuai
python-dotenv

运行方式：
```bash
streamlit run characterglm_api_demo_streamlit.py
```
"""
import os
import itertools
from io import StringIO
from typing import Iterator, Optional

import streamlit as st
from dotenv import load_dotenv

from utils import generate_role_desc

# 通过.env文件设置环境变量
# reference: https://github.com/theskumar/python-dotenv
load_dotenv()

import api
from api import generate_chat_scene_prompt, generate_role_appearance, get_characterglm_response, generate_cogview_image
from data_types import TextMsg, ImageMsg, TextMsgList, MsgList, filter_text_msg

st.set_page_config(page_title="CharacterGLM API Demo", page_icon="🤖", layout="wide")
debug = os.getenv("DEBUG", "").lower() in ("1", "yes", "y", "true", "t", "on")


def update_api_key(key: Optional[str] = None):
    if debug:
        print(f'update_api_key. st.session_state["API_KEY"] = {st.session_state["API_KEY"]}, key = {key}')
    key = key or st.session_state["API_KEY"]
    if key:
        api.API_KEY = key


# 设置API KEY
api_key = st.sidebar.text_input("API_KEY", value=os.getenv("API_KEY", ""), key="API_KEY", type="password",
                                on_change=update_api_key)
update_api_key(api_key)

# 初始化
if "history" not in st.session_state:
    st.session_state["history"] = []
if "role1_history" not in st.session_state:
    st.session_state["role1_history"] = []
if "role2_history" not in st.session_state:
    st.session_state["role2_history"] = []
if "first_question" not in st.session_state:
    st.session_state["first_question"] = "你今天还好吗?"
if "is_first_question" not in st.session_state:
    st.session_state["is_first_question"] = True
if "noval" not in st.session_state:
    st.session_state["noval"] = ""
if "meta" not in st.session_state:
    st.session_state["meta"] = {
        "user_info": "罗密欧",
        "bot_info": "",
        "bot_name": "朱丽叶",
        "user_name": "",
        "style": "二次元",
    }


def init_session():
    st.session_state["history"] = []
    st.session_state["role1_history"] = []
    st.session_state["role2_history"] = []
    st.session_state["noval"] = ""
    st.session_state["first_question"] = "你今天还好吗?"
    st.session_state["is_first_question"] = True


# 4个输入框，设置meta的4个字段
meta_labels = {
    "bot_name": "角色名",
    "user_name": "用户名",
    "bot_info": "角色人设",
    "user_info": "用户人设",
}

progress_text = "tokenize"
tokenize_bar = st.progress(0, text=progress_text)
uploaded_file = st.file_uploader("选择一个文件")


def on_upload_btn_clicked():
    print(uploaded_file)
    if uploaded_file is not None:
        st.write("请选择一本小说:", uploaded_file.name)
        st.session_state["noval"] = StringIO(uploaded_file.getvalue().decode("utf-8")).read()


st.button('阅读小说', on_click=on_upload_btn_clicked)


def on_role_1_click():
    role_1 = st.session_state["bot_name"]
    noval = st.session_state["noval"]

    role_1_desc = "".join(generate_role_desc("user", role_1, noval))
    st.session_state["meta"]["bot_info"] = role_1_desc


def on_role_2_click():
    role_2 = st.session_state["user_name"]
    noval = st.session_state["noval"]

    role_2_desc = "".join(generate_role_desc("user", role_2, noval))
    st.session_state["meta"]["user_info"] = role_2_desc


# 2x2 layout
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.text_input(label="角色名", key="bot_name", value="朱丽叶",
                      on_change=lambda: st.session_state["meta"].update(bot_name=st.session_state["bot_name"]),
                      help="模型所扮演的角色的名字，不可以为空")
        st.button(label="生成角色描述", key="role1Btn", on_click=on_role_1_click,
                  help="模型所扮演的角色的名字，不可以为空")
        st.text_area(label="角色人设", key="bot_info",
                     on_change=lambda: st.session_state["meta"].update(bot_info=st.session_state["bot_info"]),
                     help="角色的详细人设信息，不可以为空")

    with col2:
        st.text_input(label="用户名", value="罗密欧", key="user_name",
                      on_change=lambda: st.session_state["meta"].update(user_name=st.session_state["user_name"]),
                      help="用户的名字，默认为用户")
        st.button(label="生成角色描述", key="role2Btn", on_click=on_role_2_click,
                  help="模型所扮演的角色的名字，不可以为空")
        st.text_area(label="用户人设", value="", key="user_info",
                     on_change=lambda: st.session_state["meta"].update(user_info=st.session_state["user_info"]),
                     help="用户的详细人设信息，可以为空")

st.selectbox("风格", options=["二次元风格", "暗黑哥特风格", "中式古风"], key="style",
             on_change=lambda: st.session_state["meta"].update(user_info=st.session_state["style"]),
             help="风格，可以为空")


def verify_meta() -> bool:
    # 检查`角色名`和`角色人设`是否空，若为空，则弹出提醒
    if st.session_state["meta"]["bot_name"] == "" or st.session_state["meta"]["bot_info"] == "":
        st.error("角色名和角色人设不能为空")
        return False
    else:
        return True


def draw_new_image():
    """生成一张图片，并展示在页面上"""
    if not verify_meta():
        return
    text_messages = filter_text_msg(st.session_state["history"])
    if text_messages:
        # 若有对话历史，则结合角色人设和对话历史生成图片
        image_prompt = "".join(
            generate_chat_scene_prompt(
                text_messages[-10:],
                meta=st.session_state["meta"]
            )
        )
    else:
        # 若没有对话历史，则根据角色人设生成图片
        image_prompt = "".join(generate_role_appearance(st.session_state["meta"]["bot_info"]))

    if not image_prompt:
        st.error("调用chatglm生成Cogview prompt出错")
        return

    # TODO: 加上风格选项
    image_prompt = st.session_state["meta"]["style"] + image_prompt.strip()
    # image_prompt = '二次元风格。' + image_prompt.strip()

    print(f"image_prompt = {image_prompt}")
    n_retry = 3
    st.markdown("正在生成图片，请稍等...")
    for i in range(n_retry):
        try:
            img_url = generate_cogview_image(image_prompt)
        except Exception as e:
            if i < n_retry - 1:
                st.error("遇到了一点小问题，重试中...")
            else:
                st.error("又失败啦，点击【生成图片】按钮可再次重试")
                return
        else:
            break
    img_msg = ImageMsg({"role": "image", "image": img_url, "caption": image_prompt})
    # 若history的末尾有图片消息，则替换它，（重新生成）
    # 否则，append（新增）
    while st.session_state["history"] and st.session_state["history"][-1]["role"] == "image":
        st.session_state["history"].pop()
    st.session_state["history"].append(img_msg)
    st.rerun()


button_labels = {
    "clear_meta": "清空人设",
    "clear_history": "清空对话历史",
    "gen_picture": "生成图片"
}
if debug:
    button_labels.update({
        "show_api_key": "查看API_KEY",
        "show_meta": "查看meta",
        "show_history": "查看历史"
    })

# 在同一行排列按钮
with st.container():
    n_button = len(button_labels)
    cols = st.columns(n_button)
    button_key_to_col = dict(zip(button_labels.keys(), cols))

    with button_key_to_col["clear_meta"]:
        clear_meta = st.button(button_labels["clear_meta"], key="clear_meta")
        if clear_meta:
            st.session_state["meta"] = {
                "user_info": "",
                "bot_info": "",
                "bot_name": "",
                "user_name": "",
            }
            st.rerun()

    with button_key_to_col["clear_history"]:
        clear_history = st.button(button_labels["clear_history"], key="clear_history")
        if clear_history:
            init_session()
            st.rerun()

    with button_key_to_col["gen_picture"]:
        gen_picture = st.button(button_labels["gen_picture"], key="gen_picture")

    if debug:
        with button_key_to_col["show_api_key"]:
            show_api_key = st.button(button_labels["show_api_key"], key="show_api_key")
            if show_api_key:
                print(f"API_KEY = {api.API_KEY}")

        with button_key_to_col["show_meta"]:
            show_meta = st.button(button_labels["show_meta"], key="show_meta")
            if show_meta:
                print(f"meta = {st.session_state['meta']}")

        with button_key_to_col["show_history"]:
            show_history = st.button(button_labels["show_history"], key="show_history")
            if show_history:
                print(f"history = {st.session_state['history']}")

# 展示对话历史
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(msg["content"])
    elif msg["role"] == "image":
        with st.chat_message(name="assistant", avatar="assistant"):
            st.image(msg["image"], caption=msg.get("caption", None))
    else:
        raise Exception("Invalid role")

if gen_picture:
    draw_new_image()

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()


def output_stream_response(response_stream: Iterator[str], placeholder):
    content = ""
    for content in itertools.accumulate(response_stream):
        placeholder.markdown(content)
    return content


def role1_chat():
    if not verify_meta():
        return
    if not api.API_KEY:
        st.error("未设置API_KEY")

    response_stream = get_characterglm_response(filter_text_msg(st.session_state["history"]),
                                                meta=st.session_state["meta"])
    bot_response = output_stream_response(response_stream, message_placeholder)
    if not bot_response:
        message_placeholder.markdown("生成出错")
        st.session_state["history"].pop()
    else:
        st.session_state["history"].append(TextMsg({"role": "assistant", "content": bot_response}))
        st.session_state["role1_history"].append(TextMsg({"role": "assistant", "content": bot_response}))


def role2_chat():
    if not verify_meta():
        return
    if not api.API_KEY:
        st.error("未设置API_KEY")

    response_stream = get_characterglm_response(filter_text_msg(st.session_state["history"]),
                                                meta=st.session_state["meta"])
    bot_response = output_stream_response(response_stream, message_placeholder)
    if not bot_response:
        message_placeholder.markdown("生成出错")
        st.session_state["history"].pop()
    else:
        st.session_state["history"].append(TextMsg({"role": "user", "content": bot_response}))
        st.session_state["role2_history"].append(TextMsg({"role": "user", "content": bot_response}))


st.session_state["history"].append(TextMsg({"role": "user", "content": st.session_state["first_question"]}))


def save_chat():
    with open('ret.txt', 'w') as f:
        for line in st.session_state["history"]:
            f.write(f"{line}\n")


with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button(label="角色1说一句", key="role1SpeakBtn", on_click=role1_chat)

    with col2:
        st.button(label="角色2说一句", key="role2SpeakBtn", on_click=role2_chat)

    with col3:
        st.button(label="保存对话", key="saveBtn", on_click=save_chat)
