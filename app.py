import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import InferenceClient

st.set_page_config(page_title="مرشد حاسبات أسيوط الأهلية", page_icon="💻", layout="centered")

st.title("المرشد الأكاديمي الذكي 💻🤖")
st.caption("كلية الحاسبات والذكاء الاصطناعي - جامعة أسيوط الأهلية (مشروع طلابي مستقل)")

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

@st.cache_resource
def init_vectorstore():
    file_path = Path(__file__).parent / "college_knowledge_base.txt"
    
    if not file_path.exists():
        st.error(f"تعذر العثور على الملف: {file_path.name}. تأكد من رفعه في المستودع الرئيسي بنفس الاسم تماماً.")
        st.stop()

    loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n---\n", "\n## ", "\n### ", "\n\n", "\n"]
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

vectorstore = init_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def generate_response(prompt_text):
    if not HF_TOKEN:
        return "⚠️ تنبيه: مفتاح HF_TOKEN غير مضبوط في Secrets على منصة Streamlit."
    
    try:
        client = InferenceClient(api_key=HF_TOKEN)
        messages = [{"role": "user", "content": prompt_text}]
        
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=600,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"حدث خطأ أثناء التواصل مع النموذج: {str(e)}"

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "أهلاً بكم في كلية الحاسبات والذكاء الاصطناعي - جامعة أسيوط الأهلية! 💻🤖\n\n"
            "أنا مرشدكم الأكاديمي الذكي، أساعدكم في استكشاف البرامج الدراسية وشروط اللائحة وتوزيع المقررات.\n\n"
            "⚠️ **تنويه:** هذا البوت هو مجهود طلابي مستقل للمساعدة، ولا يُعتبر جهة رسمية بديلة عن إدارة الكلية وشؤون الطلاب."
        )
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("اكتب استفسارك عن المقررات أو الأقسام..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    matched_docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in matched_docs])

    prompt = f"""أنت المرشد الأكاديمي لكلية الحاسبات والذكاء الاصطناعي بجامعة أسيوط الأهلية.
أجب عن السؤال بناءً على السياق الآتي حصراً بدقة واختصار دون اختلاق معلومات:
{context}

تنبيه إلزامي: إذا كان السؤال عن مواد الترم الأول نبّه لشرط مقرر "أساسيات الرياضيات" لشعبة علمي علوم.
إذا لم تجد الإجابة في السياق، أجب بلطف بأن المعلومة غير مسجلة بالدليل المتاح وعليه مراجعة شؤون الطلاب.

السؤال: {user_input}
الإجابة:"""

    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة اللائحة والمقررات..."):
            reply = generate_response(prompt)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
