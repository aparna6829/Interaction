import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from nltk.tokenize import word_tokenize




st.set_page_config(page_icon="🎙️", page_title="Interaction", layout="wide")

header = st.container()
header.write(f"""
    <div class='fixed-header'>
        <h1>Interactive Bot</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .st-emotion-cache-vj1c9o {
            background-color: rgb(242, 242, 242, 0.68);
        }
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 0;
            background-color: rgb(242, 242, 242, 0.68);
            z-index: 999;
            text-align: center;
            margin-top: -18px;
        }
        .fixed-header {
            border-bottom: 0;
        }
        .st-emotion-cache-1rsyhoq p {
        word-break: break-word;
        border: 1px solid;
        background-color: rgb(242, 242, 242, 0.68);
        margin-top : 10px;
        border-radius : 5px;
        line-height : 2.2;
    }
    .st-emotion-cache-1vt4y43 {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: rgb(255, 255, 255);
    border: 1px solid rgb(13 13 14);
    }   

    </style>
    """,
    unsafe_allow_html=True
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

class VectorDatabase:
    def __init__(self, filename="Queries.json"):
        self.filename = filename
        self.load_vectors()

    def load_vectors(self):
        try:
            with open(self.filename, "r") as f:
                self.vectors = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.vectors = []

    def save_vectors(self):
        with open(self.filename, "w") as f:
            json.dump(self.vectors, f, indent=4)  

    def save_vector(self, question, answer):
        entry = {"question": question, "answer": answer}
        self.vectors.append(entry)
        self.save_vectors()
        print("Vector saved:", question)

    def get_vector(self, question):
        for entry in self.vectors:
            if entry["question"] == question:  # Updated key from 'query' to 'question'
                return entry["answer"]
        return None


def initialize():
    AZURE_SPEECH_KEY = st.secrets["AZURE_SPEECH_KEY"]
    AZURE_SPEECH_REGION = st.secrets["AZURE_SPEECH_REGION"]


 
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    return speech_config, audio_config

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def find_customer_details(mobile_number, json_data):
    for item in json_data:
        if isinstance(item, dict) and "mobile_number" in item:
            if item["mobile_number"].replace("-", "").strip() == mobile_number.replace("-", "").strip():
                return item
    return None

def fill_placeholders(question, customer_details):
    if customer_details:
        question = question.replace("{customer_name}", customer_details.get("customer_name", ""))
        question = question.replace("{amount_paid}", customer_details.get("amount_paid", ""))
        question = question.replace("{payment_date}", customer_details.get("payment_date", ""))
    return question

def stop_speaking_on_stop_words(synthesizer, stop_words=None):
    if stop_words is None:
        stop_words = ["stop", "please wait", "pause", "just a moment", "halt", "stay put", "one moment please", "wait a second", "wait a minute", "Telexa"]

    speech_config, audio_config = initialize()
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    def recognized_cb(evt):
        recognized_text = evt.result.text.lower()
        for word in stop_words:
            if word in recognized_text:
                st.write(f"'{word}' detected. Stopping speech.")
                stop_future = synthesizer.stop_speaking_async()
                stop_future.get()
                st.write("Go ahead.")
                synthesizer.speak_text_async("Yes! Go ahead").get()
                break

    recognizer.recognized.connect(recognized_cb)
    recognizer.start_continuous_recognition_async()

    return recognizer




def generate_response(embedding_path, query):
    vector_db = VectorDatabase()
    saved_vector = vector_db.get_vector(query)

    # If query vector is present in the database, return it as a callable
    if saved_vector is not None:
        print("Vector already present in the database:", saved_vector)
        return lambda x: {"answer": saved_vector}
    else:
        embeddings = HuggingFaceEmbeddings()

        # Load FAISS vector database
        try:
            vector = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            return None

        # Initialize the AzureChatOpenAI model
        llm = ChatOpenAI(model="gpt-4", api_key=st.secrets["OPENAI_API_KEY"], temperature=0.2, base_url=" https://api.openai.com/v1")


        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, chain_type="stuff", return_source_documents=True, retriever=vector.as_retriever()
        )
        return chain
     




def main():
    vector_db = VectorDatabase()

    speech_config, audio_config = initialize()
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)
    json_data = load_json_data("details.json")

    
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False

    customer_details = None

    col1, col2 = st.columns(2)
   
    with col1:
        if st.button("Start Listening"):
            st.session_state.is_listening = True
   
    with col2:
        if st.button("Stop Listening"):
            st.session_state.is_listening = False

    stop_word_recognizer = None

    if st.session_state.is_listening:
        st.write("You can speak now. I'm listening... Say 'stop' to pause the bot's speech.")
       
        stop_word_recognizer = stop_speaking_on_stop_words(speech_synthesizer)
        
        # Ask for mobile number
        mobile_number_question = "Can you please provide the registered mobile number of the customer for verification?"
        st.write(f"🎙️{mobile_number_question}")
        speech_synthesizer.speak_text_async(mobile_number_question).get()

        while st.session_state.is_listening:
            speech_recognition_result = speech_recognizer.recognize_once_async().get()
            mobile_number = speech_recognition_result.text.strip()

            if mobile_number:
                st.write(f"🙎🏼‍♀️User_Response: {mobile_number}")
                customer_details = find_customer_details(mobile_number, json_data)
                if customer_details:
                    st.write(f"🙎🏼‍♀️Customer details found: {customer_details}")
                    
                    questions = [
                        
                        "Thank you. I have verified the details. The customer's name is {customer_name}. Can you please provide the customer's loan account number or credit card number for further verification?",
                        "We have checked our records. A payment of {amount_paid} was made on {payment_date} for this customer.",
                        "Is there a better offer or a loan settlement option available for this customer?",
                        "Thank you for the offer. Can you please provide your mobile number so we can discuss this further?"
                    ]

                    for question in questions:
                        if not st.session_state.is_listening:
                            break

                        question_to_ask = fill_placeholders(question, customer_details)
                        st.write(f"🎙️{question_to_ask}")
                        speech_synthesizer.speak_text_async(question_to_ask).get()

                        speech_recognition_result = speech_recognizer.recognize_once_async().get()
                        input_text = speech_recognition_result.text.strip()
                        if input_text:
                            st.write(f"🙎🏼‍♀️User_Response: {input_text}")

                    st.write("Now, you can ask your queries.")
                    text_2 = "Now, you can ask your queries."
                    speech_synthesizer.speak_text_async(text_2).get()
                    
                    while st.session_state.is_listening:
                        speech_recognition_result = speech_recognizer.recognize_once_async().get()
                        input_text = speech_recognition_result.text.strip()
                        if input_text:
                            st.write(f"🙎🏼‍♀️Query_asked: {input_text}")
                            with st.spinner("Going to Find the Response for your query"):
                                text_1 = "Going to Find the Response for your query"
                                speech_synthesizer.speak_text_async(text_1).get()
                                query = input_text
                           
                                chain = generate_response("Question_Answer_INDEX", input_text)
                                result = chain({"question": query})
                                response = result['answer']
                               

                                # Save the response vector in the vector database
                                vector_db.save_vector(input_text,response)

                                
                            st.write(f"🎙️{response}")
                            speech_synthesizer.speak_text_async(response).get()
    
                else:
                    st.write("No customer details found for the provided mobile number.")
                
                break
        else:
            st.write("Click 'Start Listening' to begin.")

    if stop_word_recognizer:
        stop_word_recognizer.stop_continuous_recognition_async()

if __name__ == '__main__':
    main()
