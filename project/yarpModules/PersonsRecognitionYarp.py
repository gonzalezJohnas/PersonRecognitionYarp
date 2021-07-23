import sys
import os
import time

from torchvision import transforms
import torch, torchaudio
import yarp
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from project.voiceRecognition.speaker_embeddings import SpeakerEmbeddings
from project.faceRecognition.utils import format_face_coord, face_alignement, format_names_to_bottle, fixed_image_standardization
from project.AVRecognition.lit_AVperson_classifier import LitSpeakerClassifier, Backbone

import scipy.io.wavfile as wavfile
import scipy
import dlib
import cv2 as cv

def info(msg):
    print("[INFO] {}".format(msg))


class PersonsRecognition(yarp.RFModule):
    """
    Description:
        Class to recognize speaker from the audio

    Args:
        input_port  : Audio from remoteInterface

    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # handle port for the RFModule
        self.module_name = None
        self.handle_port = None


        # Define vars to receive audio
        self.audio_in_port = None
        self.audio_power_port = None

        # Define port to control the head motion
        self.label_outputPort = None

        # Speaker module parameters
        self.model_audio = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.dataset_path = None
        self.db_embeddings_audio = None
        self.process = True
        self.threshold_audio = None
        self.threshold_power = None
        self.length_input = None
        self.resample_trans = None
        self.speaker_emb = []

        # Parameters for the audio
        self.sound = None
        self.audio = []
        self.np_audio = None
        self.nb_samples_received = 0
        self.sampling_rate = None

        # Define  port to receive an Image
        self.image_in_port = yarp.BufferedPortImageRgb()
        self.face_coord_port = yarp.BufferedPortBottle()

        self.detector = dlib.get_frontal_face_detector()

        # Image parameters
        self.width_img = None
        self.height_img = None
        self.input_img_array = None
        self.frame = None
        self.coord_face = None
        self.threshold_face = None
        self.face_emb = []


        # Model face recognition modele
        self.modele_face = None
        self.db_embeddings_face = None
        self.trans = None

        # Model for cross-modale recognition
        self.model_av = None
        self.sm = torch.nn.Softmax(dim=1)
        self.threshold_multimodal = None

        self.device = None


    def configure(self, rf):

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Define vars to receive audio
        self.audio_in_port = yarp.BufferedPortSound()

        self.label_outputPort = yarp.BufferedPortBottle()
        self.audio_power_port = yarp.Port()

        # Module parameters
        self.module_name = rf.check("name",
                                    yarp.Value("PersonRecognition"),
                                    "module name (string)").asString()
        self.dataset_path = rf.check("dataset_path",
                                   yarp.Value(
                                       ""),
                                   "Root path of the embeddings database (voice & face) (string)").asString()

        self.length_input = rf.check("length_input",
                                     yarp.Value(1),
                                     "length audio input in seconds (int)").asInt()

        self.threshold_audio = rf.check("threshold_audio",
                                  yarp.Value(0.42),
                                  "threshold_audio for detection (double)").asDouble()

        self.threshold_face = rf.check("threshold_face",
                                        yarp.Value(0.42),
                                        "threshold_audio for detection (double)").asDouble()

        self.threshold_power = rf.check("threshold_vad",
                                        yarp.Value(1.5),
                                        "threshold for VAD detection (double)").asDouble()

        self.sampling_rate = rf.check("fs",
                                      yarp.Value(48000),
                                      " Sampling rate of the incoming audio signal (int)").asInt()

        self.resample_trans = torchaudio.transforms.Resample(self.sampling_rate, 16000)


        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Load Database  for audio embeddings
        try:
            self.db_embeddings_audio = SpeakerEmbeddings(os.path.join(self.dataset_path, "audio"))
        except FileNotFoundError:
            info(f"Unable to find dataset {SpeakerEmbeddings(os.path.join(self.dataset_path, 'audio'))}")


        # Audio and power
        self.audio_in_port.open('/' + self.module_name + '/audio:i')
        self.audio_power_port.open('/' + self.module_name + '/power:i')
        # Label
        self.label_outputPort.open('/' + self.module_name + '/label:o')

        # Image anf face
        self.width_img = rf.check('width', yarp.Value(320),
                                  'Width of the input image').asInt()

        self.height_img = rf.check('height', yarp.Value(244),
                                   'Height of the input image').asInt()
        self.face_coord_port.open('/' + self.module_name + '/coord:i')
        self.image_in_port.open('/' + self.module_name + '/image:i')
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()

        # Modele for  face embeddings
        self.trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
            transforms.Resize((180, 180))
        ])
        try:
            self.modele_face = torch.load("/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/faceRecognition/saved_model/model_triple_facerecogntion_66%.pt")
            self.db_embeddings_face = SpeakerEmbeddings(os.path.join(self.dataset_path, "face"))
        except FileNotFoundError:
            info(f"Unable to find dataset {SpeakerEmbeddings(os.path.join(self.dataset_path, 'face'))}")

        self.modele_face.eval()

        self.model_av = torch.load(
            "/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/AVRecognition/model_audiovisual.pt")
        self.model_av.eval()
        self.threshold_multimodal = 0.8

        info("Initialization complete")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        return True


    def interruptModule(self):
        print("[INFO] Stopping the module")
        self.audio_in_port.interrupt()
        self.label_outputPort.interrupt()
        self.audio_power_port.interrupt()
        self.handle_port.interrupt()
        self.image_in_port.interrupt()
        self.face_coord_port.interrupt()

        return True

    def close(self):
        self.audio_in_port.close()
        self.handle_port.close()
        self.label_outputPort.close()
        self.image_in_port.close()
        self.audio_power_port.close()
        self.face_coord_port.close()

        return True

    def respond(self, command, reply):
        ok = False

        # Is the command recognized
        rec = False

        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False

        elif command.get(0).asString() == "start":
            reply.addString("ok")
            self.process = True

        elif command.get(0).asString() == "stop":
            self.process = False
            reply.addString("ok")

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                self.threshold_audio = command.get(2).asDouble()
                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                reply.addDouble(self.threshold_audio)
            else:
                reply.addString("nack")

        else:
            reply.addString("nack")

        return True

    def getPeriod(self):
        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.05

    def record_audio(self):
        self.sound = self.audio_in_port.read(False)
        if self.sound:

            chunk = np.zeros((self.sound.getChannels(), self.sound.getSamples()), dtype=np.float32)
            self.nb_samples_received += self.sound.getSamples()
            for c in range(self.sound.getChannels()):
                for i in range(self.sound.getSamples()):
                    chunk[c][i] = self.sound.get(i, c) / 32768.0

            self.audio.append(chunk)

            return True
        return False

    def read_image(self):
        input_yarp_image = self.image_in_port.read(False)

        if input_yarp_image:
            input_yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)
            self.frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                (self.height_img, self.width_img, 3)).copy()

            return True

        return False

    def get_power(self):
        max_power = 0.0
        if self.audio_power_port.getInputCount():

            power_matrix = yarp.Matrix()
            self.audio_power_port.read(power_matrix)
            power_values = [power_matrix[0, 1], power_matrix[0, 0]]
            max_power = np.max(power_values)
            # info("Max power is {}".format(max_power))

        return max_power

    def get_face_coordinate(self):
        if self.face_coord_port.getInputCount():
            self.coord_face = self.face_coord_port.read(False)
            return self.coord_face is not None

        self.coord_face = None
        return False


    def updateModule(self):
        if self.process:
            audio_power = self.get_power()
            if audio_power > self.threshold_power and self.record_audio():

                if self.nb_samples_received >= self.length_input * self.sound.getFrequency():
                    audio_signal = self.format_signal(self.audio)
                    # wavfile.write("/tmp/recording.wav", self.sound.getFrequency(), audio_signal)
                    self.speaker_emb = self.get_audio_embeddings(audio_signal)
                    self.audio = []
                    self.nb_samples_received = 0

            if self.read_image() and self.get_face_coordinate():
                try:
                    self.coord_face = format_face_coord(self.coord_face)
                    face_images = [face_alignement(f, self.frame) for f in self.coord_face]
                    self.face_emb = self.get_face_embeddings(face_images)
                except Exception:
                    info("Exception while computing face embeddings")

            if len(self.speaker_emb) and len(self.face_emb):
                person_name, score = self.predict_multimodal(self.speaker_emb, self.face_emb)
                if score > self.threshold_multimodal:
                    print(f"Recognized person with AV model {person_name}")
                    self.speaker_emb = []
                    self.face_emb = []
                    self.write_label(person_name, score, 2)
                    self.process = False

            elif len(self.speaker_emb):
                speaker_name, score = self.predict_speaker(self.speaker_emb)
                print(f"Recognized speaker {speaker_name}")
                self.speaker_emb = []
                self.write_label(speaker_name, score, 1)

            elif len(self.face_emb):
                faces_name, scores = self.predict_face(self.face_emb)
                self.face_emb = []
                print(f"Recognized faces {faces_name}")
                # for name, score in zip(faces_name, scores):
                #     self.write_label(name, score, 0)
            else:
                pass

        return True

    def format_signal(self, audio_list_samples):
        """
        Format an audio given a list of samples
        :param audio_list_samples:
        :return: numpy array
        """
        np_audio = np.concatenate(audio_list_samples, axis=1)
        np_audio = np.squeeze(np_audio)
        signal = np.transpose(np_audio, (1, 0))

        return signal

    def get_audio_embeddings(self, audio):
        resample_audio = self.resample_trans(torch.from_numpy(audio.transpose()))
        embedding = self.model_audio.encode_batch(resample_audio)

        return embedding
    
    def get_face_embeddings(self, images):
        face_embeddings = []
        with torch.no_grad():
            for np_img in images:
                cv.cvtColor(np_img, cv.COLOR_RGB2BGR, np_img)
                input_img = self.trans(np_img)
                input_img = input_img.unsqueeze_(0)
                input = input_img.to(self.device)
                emb = self.modele_face(input)
                face_embeddings.append(emb.cpu())

        return face_embeddings

    def predict_speaker(self, embedding):

        score, speaker_id = self.db_embeddings_audio.get_speaker(embedding)

        # print(f"Cosine distance for audio embeddings {score}")
        if score > self.threshold_audio:
            speaker_name = self.db_embeddings_audio.get_name_speaker(speaker_id)
        else:
            speaker_name = "Unknown"

        return speaker_name, float(score)

    def predict_face(self, embeddings):
        predicted_faces = []
        score_faces = []
        for emb in embeddings:
            score, face_id = self.db_embeddings_face.get_speaker(emb)
            # print(f"Cosine distance for face embeddings {score}")
            if score > self.threshold_face:
                face_name = self.db_embeddings_face.get_name_speaker(face_id)
            else:
                face_name = "Unknown"
            predicted_faces.append(face_name)
            score_faces.append(float(score))

        return predicted_faces, score_faces

    def predict_multimodal(self, audio_emb, face_emb):
        if audio_emb.shape[0] > 1:
            audio_emb = audio_emb[0]

        input_emb = np.hstack((audio_emb, face_emb[0]))
        with torch.no_grad():
            input_emb = torch.from_numpy(input_emb).cuda()
            outputs = self.sm(self.model_av(input_emb))

            proba, p_id = torch.max(outputs, 1)
            prediction_id = int(p_id.cpu().numpy()[0])
            score = float(proba.cpu().numpy()[0])
            recognized_name = self.db_embeddings_face.get_name_speaker(prediction_id)

        return recognized_name, score

    def write_label(self, name_speaker, score, mode):
        if self.label_outputPort.getOutputCount():
            name_bottle = self.label_outputPort.prepare()
            name_bottle.clear()
            name_bottle.addString(name_speaker)
            name_bottle.addFloat32(score)
            name_bottle.addInt(mode)

            self.label_outputPort.write()


if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        info("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    speaker_recognition = PersonsRecognition()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('peopleRecognition')
    rf.setDefaultConfigFile('peopleRecognition.ini')

    if rf.configure(sys.argv):
        speaker_recognition.runModule(rf)

    speaker_recognition.close()
    sys.exit()
