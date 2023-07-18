import streamlit as st
import  numpy as np
import os
import joblib
import pickle
from PIL import Image

st.set_page_config(
    page_title="Web Prediksi Stroke",
    layout="wide"
)
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
    }
    .medium-font{
        font-size:30px !important;
    }
    </style>
  """, unsafe_allow_html=True)
form = st.sidebar.form("form", clear_on_submit=False)
with form:        
        st.subheader("Masukan Data Sesuai Diri Anda")

        gender = st.radio(
                "Jenis Kelamin",
            ('Pria', 'Wanita')
        )

        age = st.number_input(label='Umur',
            min_value = 1,
            max_value = 85,
            value=50,
            step=1)

        hypertension = st.radio(
                "Menderita Hipertensi?",
            ('Yes', 'No'),
            help='History of hypertension symptoms or high blood pressure'
        )

        heart_disease = st.radio(
                "Menderita Penyakit Jantung?",
            ('Yes', 'No'),
            help='History of heart disease'
        )

        ever_married = st.radio(
                "Status Menikah",
            ('Menikah/Pernah Menikah', 'Tidak Menikah'),
            help = 'Has the patient ever been married?'
        )

        work_type = st.selectbox(
            label='Jenis pekerjaan',
            options=( 'Pekerja Pemerintahan','Pekerja Swasta','Wirawasta/Wirausaha','Tidak Bekerja')
            )
        
        Is_Urban_Residence = st.radio(
                "Lokasi Tempat Tinggal",
            ('Perkotaan', 'Pedesaan')
        )

        avg_glucose_level = st.slider(
            label = "Nilai Gula Darah",
            min_value = 50.0,
            max_value = 260.0,
            value=130.0,
            step=0.1
        )

        bmi = st.number_input(
            label = "Body-Mass Index (BMI)",
            min_value = 10.0,
            max_value = 55.0,
            value=25.0,
            step=1.0
        )

        smoking_status = st.selectbox(
            label='Riwayat Merokok',
            options=('Mantan Perokok','Tidak Merokok' ,'Perokok','Unknown')
            )

        submit = st.form_submit_button("Jalankan Prediksi")
    
if submit==False:
    image = Image.open('image/stroke.png')
    st.image(image)
   
  
    st.markdown('<p class="big-font">Website Prediksi Resiko Penyakit Stroke</p>', unsafe_allow_html=True)
    st.info("Mohon masukan data pada kolom disamping yang sesuai dengan keadaan anda saat ini!")
        
if submit == True:
    if gender == 'Pria':
        gender = 1
    else:
        gender = 0

    if hypertension == 'Yes':
        hypertension = 1
    else:
        hypertension = 0

    if heart_disease == 'Yes':
        heart_disease = 1
    else:
        heart_disease = 0

    if ever_married == 'Menikah/Pernah Menikah':
        ever_married = 1
    else:
        ever_married = 0

    if Is_Urban_Residence == 'Perkotaan':
        Is_Urban_Residence = 1
    else:
        Is_Urban_Residence = 0

    if work_type == 'Self-Employed':
         work_type=3
           
    elif work_type == 'Government Job':
        work_type=1

    elif work_type == 'Private Industry':
        work_type=2
    else:
        work_type=0

    if smoking_status == 'Tidak Merokok':
            smoking_status=2
           
    elif smoking_status == 'Mantan Perokok':
            smoking_status=1
    elif smoking_status == 'Perokok':
            smoking_status=3
    else:
            smoking_status=0

    current_directory = os.path.dirname(os.path.abspath(__file__))

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Is_Urban_Residence,
                    avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join(current_directory,'models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    xs=scaler.transform(x)

    model_path=os.path.join(current_directory,'models/dt20.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(xs)

    if Y_pred==0:
        st.info('Berdasarkan data yang anda masukan anda tidak berpotensi stroke')
        image = Image.open('image/sehat.jpg')
        st.image(image,width=500)
        st.markdown('<p class="medium-font">Pertahankan! dan jaga pola hidup anda</p>', unsafe_allow_html=True)
    else:
        st.error('Berdasarkan data yang anda masukan anda berpotensi stroke')
        image = Image.open('image/stroke1.jpg')
        st.image(image,width=500)
        st.markdown('<p class="medium-font">Lakukan langkah berikut untuk menurunkan resiko stroke</p>', unsafe_allow_html=True)
        st.markdown(
             '<ul><li>Periksakan diri ke dokter</li></ul>'
             '<ul><li>Atur Pola Makan Anda</li></ul>'
             '<ul><li>Rajin Berolahraga</li></ul>',
             unsafe_allow_html=True)