import streamlit as st
import pandas as pd
import pickle
from PIL import Image
image_improve=Image.open('improve.jpeg')
image_ok = Image.open('ok.jpeg')
image_worse=Image.open('worse.jpeg')
st.title('Does music enhance or disrupt your well-being?🤨')



def user_input_features():

    st.subheader('Tell me about you')
    age=st.slider('Your Age', 10.0,90.0, 25.0)
    streaming=st.selectbox('Primary Streaming service?',('Spotify', 'Pandora', 'YouTube Music','I do not use a streaming service.', 'Apple Music',
                            'Other streaming service'))
   

    
    st.subheader('How would you rate your mental health on a scale, between 0-10?')
    anxiety=st.number_input('Anxiety', min_value=0.0,max_value=10.0)
    depression=st.number_input('Depression', min_value=0.0,max_value=10.0)
    insomnia=st.number_input('Insomnia', min_value=0.0,max_value=10.0)
    ocd=st.number_input('OCD', min_value=0.0,max_value=10.0)

    st.subheader("What are your experiences with music?")
    hours=st.slider('How many hours do you listen to music per day?', 0.0,24.0, 7.0)
    working=st.selectbox('Do you listen to music at work or while studying?',('Yes', 'No'))
    instrumentalist=st.selectbox('Do you play an instrument regularly?',('Yes', 'No'))            
    composer=st.selectbox('Do you compose music?',('Yes', 'No'))
    exploratory=st.selectbox('Do you actively explore new artists/genres?',('Yes', 'No'))
    language=st.selectbox('Do you regularly listen to music with lyrics in a language you are not fluent in?',('Yes', 'No'))    
    bpm=st.number_input('Beats per minute of favorite genre')


    st.subheader("And lastly, what do you like to listen to?")
    classical=st.selectbox('Classical Musica',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    country=st.selectbox('Country',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    edm=st.selectbox('Electronic Dance Music',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    folk=st.selectbox('Folk',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    gospel=st.selectbox('Gospel',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    hiphop=st.selectbox('Hip-Hop',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    jazz=st.selectbox('Jazz',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    kpop=st.selectbox('K pop',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    latin=st.selectbox('Latin',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    lofi=st.selectbox('Lo-Fi',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    metal=st.selectbox('Metal',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    pop=st.selectbox('Pop',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    reb=st.selectbox('R&B',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    rap=st.selectbox('Rap',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    rock=st.selectbox('Rock',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    vgm=st.selectbox('Video Game Music',( 'Never','Rarely', 'Sometimes', 'Very frequently'))
    data= {
            'Age':age, 
            'Primary streaming service':streaming,
            "Hours per day":hours,
            'While working':working,
            'Instrumentalist':instrumentalist,                
            'Composer':composer,
            'Exploratory':exploratory,
            'Foreign languages':language,
            'BPM':bpm,
            'Frequency [Classical]':classical,
            'Frequency [Country]':country,
            'Frequency [EDM]':edm,
            'Frequency [Folk]':folk,
            'Frequency [Gospel]':gospel,
            'Frequency [Hip hop]':hiphop,
            'Frequency [Jazz]':jazz,
            'Frequency [K pop]':kpop,
            'Frequency [Latin]':latin,
            'Frequency [Lofi]':lofi,
            'Frequency [Metal]':metal,
            'Frequency [Pop]':pop,
            'Frequency [R&B]':reb,
            'Frequency [Rap]':rap,
            'Frequency [Rock]':rock,
            'Frequency [Video game music]':vgm,
            'Anxiety':anxiety,
            'Depression':depression,
            'Insomnia':insomnia,
            'OCD':ocd} 
                    
    features = pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.write(df)


if st.button('Predict'):
    pipeline = pickle.load(open('preprocessor.pkl','rb'))
    model=pickle.load(open('model.pkl','rb'))

    X=pipeline.transform(df)
    predict=model.predict(X)   


    if predict == 0:
         st.title("I don't know what you're listening to, but it appears to be genuinely beneficial for you.")
         st.image(image_improve)
    elif predict == 1:
         st.title("Your relationship with music seems to be quite consistent. It doesn't seem to have a significant impact on your health either way.")
         st.image(image_ok)
    else:
         st.title("Please pause for a moment. I believe it's time to consider making some adjustments to your music listening habits. It could be more beneficial for your overall well-being.")
         st.image(image_worse)

st.caption('Dataset source: https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results')