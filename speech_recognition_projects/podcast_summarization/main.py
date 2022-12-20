from helpers import *
import streamlit as st
import json

st.title("Welcome to my application that creates podcast summaries")
episode_id = st.sidebar.text_input('Please input an episode id')
button = st.sidebar.button('Get podcast summary!', on_click=save, args=(episode_id,))

if button:
    filename = FILES_FOLDER_NAME + '/' + episode_id + '_chapters.json'
    with open(filename, 'r') as _file:
        data = json.load(_file)

        chapters = data['chapters']
        podcast_title = data['podcast_title']
        episode_title = data['episode_title']
        episode_thumbnail = data['episode_thumbnail']

    st.header(f'{podcast_title} - {episode_title}')
    st.image(episode_thumbnail)

    for chapter in chapters:
        with st.expander(chapter['gist'] + '-' + get_clean_time(chapter['start'])):
            st.write(chapter['summary'])

# save("5d69d283a8a34b81b95b4b128e49eff9")
