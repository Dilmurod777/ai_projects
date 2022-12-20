from pydub import AudioSegment

audio = AudioSegment.from_wav("files/output.wav")

# increase the volume by 6 dB
audio = audio + 6
audio = audio * 2
audio = audio.fade_in(2000)
audio.export("files/mashup_output.mp3", format="mp3")

audio2 = AudioSegment.from_mp3("files/mashup_output.mp3")
print("Done")