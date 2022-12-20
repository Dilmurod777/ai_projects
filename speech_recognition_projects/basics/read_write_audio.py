import wave

if __name__ == "__main__":
    obj = wave.open("files/output.wav", "rb")

    print("Number of channels:", obj.getnchannels())
    print("Sample width:", obj.getsampwidth())
    print("Frame rate:", obj.getframerate())
    print("Number of frames:", obj.getnframes())
    print("Parameters:", obj.getparams())
    print("Duration:", obj.getnframes() / obj.getframerate())

    frames = obj.readframes(-1)
    print(f"Type of frames: {type(frames)}")
    print(f"Type of first frame: {type(frames[0])}")
    print(f"Total number of frames (with {obj.getsampwidth()} bytes per frame): {len(frames)}")

    new_obj = wave.open("files/new_output.wav", "wb")

    new_obj.setnchannels(obj.getnchannels())
    new_obj.setsampwidth(obj.getsampwidth())
    new_obj.setframerate(obj.getframerate())
    new_obj.writeframes(frames)
    new_obj.close()
