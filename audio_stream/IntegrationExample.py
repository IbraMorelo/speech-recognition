import SpeechRecognition as sr

IntSR = sr.SpeechRecognition()

with IntSR:
	while True:
		x = IntSR.get_record()
		if x:
			print('COMANDO', x)


