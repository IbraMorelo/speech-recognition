import SpeechRecognition as sr
from unittest import TestCase
from scipy.io import wavfile

class SpeechRecognitionTest(TestCase):
	def test_enqueue_frames(self):
		init_sr = sr.SpeechRecognition()
		expected_output = 2
		init_sr.enqueue_frames('One Stream')
		init_sr.enqueue_frames('Two Stream')
		self.assertEqual(len(init_sr.frames), expected_output)

	def test_waveform_downsampling(self):
		init_sr = sr.SpeechRecognition()
		original_audio = wavfile.read('audio_test/first_audio.wav')[1]
		resample_audio = init_sr.waveform_downsampling(original_audio).shape[0]
		self.assertTrue(original_audio.shape[0] > resample_audio)

	def test_waveform_complete_if_needed(self):
		init_sr = sr.SpeechRecognition()
		expected_output = 16000
		original_audio = wavfile.read('audio_test/first_audio.wav')[1]
		resample_audio = init_sr.waveform_downsampling(original_audio)
		padded_audio = init_sr.waveform_complete_if_needed(resample_audio)
		self.assertEqual(padded_audio.shape[0], 16000)

	def test_get_spectrogram(self):
		init_sr = sr.SpeechRecognition()
		expected_output = (129, 124)
		waveform_original = wavfile.read('audio_test/first_audio.wav')[1]
		waveform = init_sr.process_audio_data(waveform_original)
		spectrogram = init_sr.get_spectrogram(waveform)
		self.assertEqual(spectrogram.shape[0], expected_output[0])
		self.assertEqual(spectrogram.shape[1], expected_output[1])

	def test_init_and_exit(self):
		init_sr = sr.SpeechRecognition()
		with init_sr:
			self.assertTrue(init_sr.stream.is_active())

	def test_run_inference(self):
		init_sr = sr.SpeechRecognition()
		expected_output = 'derecha'
		waveform_original = wavfile.read('audio_test/second_audio.wav')[1]
		spectrogram = init_sr.get_spectrogram(waveform_original)
		command = init_sr.run_inference(spectrogram)
		self.assertEqual(command, expected_output)

	def test_get_record(self):
		init_sr = sr.SpeechRecognition()
		expected_output = ''
		init_sr.enqueue_frames([1])
		init_sr.enqueue_frames([2])
		self.assertFalse(init_sr.get_record())