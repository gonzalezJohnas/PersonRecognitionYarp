import sys
import os
import time

from torchvision import transforms
import torch, torchaudio
import yarp
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from project.voiceRecognition.speaker_embeddings import SpeakerEmbeddings
from project.faceRecognition.utils import format_face_coord, face_alignement, format_names_to_bottle, \
    fixed_image_standardization, get_center_face
from project.AVRecognition.lit_AVperson_classifier import LitSpeakerClassifier, Backbone
from project.yarpModules.DatabaseHandler import DatabaseHandler

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

        # Predictions parameters
        self.label_outputPort = None
        self.predictions = []
        self.database = None

        # Speaker module parameters
        self.model_audio = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.dataset_path = None
        self.db_embeddings_audio = None
        self.process = False
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
        self.opc_port = yarp.RpcClient()

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
        self.faces_img = []
        self.face_coord_request = None
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
        self.label_outputPort = yarp.Port()
        self.audio_power_port = yarp.BufferedPortBottle()


        # Module parameters
        self.module_name = rf.check("name",
                                    yarp.Value("PersonRecognition"),
                                    "module name (string)").asString()
        self.dataset_path = rf.check("dataset_path",
                                     yarp.Value(
                                         ""),
                                     "Root path of the embeddings database (voice & face) (string)").asString()

        self.database = DatabaseHandler(self.dataset_path)

        self.length_input = rf.check("length_input",
                                     yarp.Value(1),
                                     "length audio input in seconds (int)").asInt()

        self.threshold_audio = rf.check("threshold_audio",
                                        yarp.Value(0.32),
                                        "threshold_audio for detection (double)").asDouble()

        self.threshold_face = rf.check("threshold_face",
                                       yarp.Value(0.55),
                                       "threshold_face for detection (double)").asDouble()

        self.threshold_power = rf.check("threshold_vad",
                                        yarp.Value(18.5),
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
        self.face_coord_port.setStrict(False)

        self.image_in_port.open('/' + self.module_name + '/image:i')
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()

        self.opc_port.open('/' + self.module_name + '/OPC:rpc')
        # Modele for  face embeddings
        self.trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
            transforms.Resize((180, 180))
        ])
        try:
            self.modele_face = torch.load(
                "/home/icub/PycharmProjects/SpeakerRecognitionYarp/project/faceRecognition/saved_model/model_triple_facerecogntion_66%.pt")
            self.db_embeddings_face = SpeakerEmbeddings(os.path.join(self.dataset_path, "face"))
        except FileNotFoundError:
            info(f"Unable to find dataset {SpeakerEmbeddings(os.path.join(self.dataset_path, 'face'))}")

        self.modele_face.eval()
        # self.db_embeddings_face.create_embeddings(self.modele_face)

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

        elif command.get(0).asString() == "save":
            if command.get(1).asString() == "face":
                name = command.get(2).asString().lower()
                if name in self.db_embeddings_face.data_dict.keys():
                    self.db_embeddings_face.data_dict[name] = self.db_embeddings_face.data_dict[name] + self.face_emb
                else:
                    self.db_embeddings_face.data_dict[name] = self.face_emb

                self.database.save_faces(self.faces_img, self.face_emb, name)
                self.faces_img = []
                self.face_emb = []

            reply.addString("ok")

        elif command.get(0).asString() == "reset":
            self.db_embeddings_face.excluded_faces = []

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                self.threshold_audio = command.get(2).asDouble()
                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                reply.addDouble(self.threshold_audio)
            elif command.get(1).asString() == "face":
                self.face_coord_request = [command.get(2).asDouble(), command.get(3).asDouble(), command.get(4).asDouble(),
                                 command.get(5).asDouble()]

                reply.addString("ok")

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
        max_power = -1
        if self.audio_power_port.getInputCount():
            power_matrix = self.audio_power_port.read(False)
            if power_matrix:

                power_matrix = power_matrix.get(2).asList()
                power_values = [power_matrix.get(0).asDouble(), power_matrix.get(1).asDouble()]
                max_power = np.max(power_values)
        return max_power

    def get_face_coordinate(self):
        if self.face_coord_port.getInputCount():
            self.coord_face = self.face_coord_port.read(False)
            return self.coord_face is not None

        self.coord_face = None
        return False

    def get_name_in_memory(self):
        if self.opc_port.getOutputCount():
            reply = yarp.Bottle()
            cmd = yarp.Bottle("ask")
            list_condition = cmd.addList()
            cond1 = list_condition.addList()
            cond1.addString("verified")
            cond1.addString("==")
            cond1.addInt(1)

            self.opc_port.write(cmd, reply)
            list_id = reply.get(1).asList().get(1).asList()

            reply_id = yarp.Bottle()
            for i in range(list_id.size()):
                cmd_str = "get ((id " + str(list_id.get(i).asInt()) + ") (propSet (label_tracker)))"
                cmd = yarp.Bottle(cmd_str)
                self.opc_port.write(cmd, reply_id)
                name = reply_id.get(1).asList().get(0).asList().get(1).asString()
                self.db_embeddings_face.excluded_faces.append(name)

    def updateModule(self):
        current_face_emb = []

        self.read_image()
        audio_power = self.get_power()
        self.record_audio()
        self.get_name_in_memory()
        self.get_face_coordinate()
        if self.process:

            if audio_power > self.threshold_power and self.nb_samples_received >= self.length_input * self.sound.getFrequency():
                    audio_signal = self.format_signal(self.audio)
                    self.speaker_emb = self.get_audio_embeddings(audio_signal)
                    self.audio = []
                    self.nb_samples_received = 0

            if self.coord_face:
                try:
                    self.coord_face = format_face_coord(self.coord_face)
                    self.coord_face = get_center_face(self.coord_face, self.width_img)
                    face_img = [face_alignement(f, self.frame) for f in self.coord_face]
                    current_face_emb = self.get_face_embeddings(face_img)

                    self.faces_img = self.faces_img + face_img
                    self.face_emb.append(current_face_emb[0].numpy())

                except Exception as e:
                    info("Exception while computing face embeddings" + str(e))

            # if len(self.speaker_emb) and len(self.face_emb):
            #     person_name, score = self.predict_multimodal(self.speaker_emb, self.face_emb)
            #     if score > self.threshold_multimodal:
            #         print(f"Recognized person with AV model {person_name}")
            #         self.speaker_emb = []
            #         self.face_emb = []
            #         self.write_label(person_name, score, 2)
            #         self.process = False

            if len(self.speaker_emb):
                speaker_name, score = self.predict_speaker(self.speaker_emb)
                print(f"Recognized speaker {speaker_name}")
                self.speaker_emb = []
                self.write_label(speaker_name, score, 1)

            elif len(current_face_emb):
                faces_name, scores = self.predict_face(current_face_emb)
                print(f"Recognized faces {faces_name}")
                for name, score in zip(faces_name, scores):
                    self.write_label(name, score, 0)

        elif self.face_coord_request is not None:
            face_img = [face_alignement(self.face_coord_request, self.frame)]
            face_emb = self.get_face_embeddings(face_img)
            faces_name, scores = self.predict_face(face_emb)
            self.write_label(faces_name[0], scores[0], 1)
            self.face_coord_request = None

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
        """
        Generate voice embedding from audio sample
        :param audio:
        :return:
        """
        resample_audio = self.resample_trans(torch.from_numpy(audio.transpose()))
        embedding = self.model_audio.encode_batch(resample_audio)

        return embedding

    def get_face_embeddings(self, images):
        """
       Generate faces embedding from images of faces
       :param images: list of cropped faces (list->np.array)
       :return:  (list->np.array)
        """
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

        score, speaker_name = self.db_embeddings_audio.get_speaker_db_scan(embedding)

        # print(f"Cosine distance for audio embeddings {score}")
        if score == -1:
            speaker_name = "unknown"

        return speaker_name, float(score)

    def predict_face(self, embeddings):
        predicted_faces = []
        score_faces = []
        for emb in embeddings:
            score, face_name = self.db_embeddings_face.get_speaker_db_scan(emb)
            if score == -1:
                face_name = "unknown"

            predicted_faces.append(face_name)
            score_faces.append(score)
        self.db_embeddings_face.excluded_faces = []
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
            name_bottle = yarp.Bottle()
            name_bottle.clear()
            name_bottle.addString(name_speaker)
            name_bottle.addFloat32(score)
            name_bottle.addInt(mode)

            self.label_outputPort.write(name_bottle)


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
