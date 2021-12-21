import sys
import os
import time

from torchvision import transforms
import torch, torchaudio
import yarp
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from project.voiceRecognition.speaker_embeddings import EmbeddingsHandler
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
        Class to recognize a person from the audio or the face

    Args:
        input_port  : Audio from remoteInterface, raw image from iCub cameras

    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # handle port for the RFModule
        self.module_name = None
        self.handle_port = None
        self.process = False

        # Define vars to receive audio
        self.audio_in_port = None
        self.eventPort = None
        self.is_voice = False

        # Predictions parameters
        self.label_outputPort = None
        self.predictions = []
        self.database = None

        # Speaker module parameters
        self.model_audio = None
        self.dataset_path = None
        self.db_embeddings_audio = None
        self.threshold_audio = None
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

        # Port to query and update the memory (OPC)
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
        self.face_model_path = None

        # Model for cross-modale recognition
        self.model_av = None
        self.sm = torch.nn.Softmax(dim=1)
        self.threshold_multimodal = None

        self.device = None

        self.save_face = False
        self.name = ""
        self.predict = False

    def configure(self, rf):

        success = True

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Define vars to receive audio
        self.audio_in_port = yarp.BufferedPortSound()
        self.label_outputPort = yarp.Port()
        self.eventPort = yarp.BufferedPortBottle()

        # Module parameters
        self.module_name = rf.check("name",
                                    yarp.Value("PersonRecognition"),
                                    "module name (string)").asString()
        self.handle_port.open('/' + self.module_name)

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

        self.face_model_path = rf.check("face_model_path",
                                        yarp.Value(""),
                                        "Path of the model for face embeddings (string)").asString()

        # Set the device for inference for the models
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        success &= self.load_model_face()

        self.sampling_rate = rf.check("fs",
                                      yarp.Value(48000),
                                      " Sampling rate of the incoming audio signal (int)").asInt()

        success &= self.load_model_audio()

        # Audio and voice events
        self.audio_in_port.open('/' + self.module_name + '/audio:i')
        self.eventPort.open('/' + self.module_name + '/events:i')

        # Label
        self.label_outputPort.open('/' + self.module_name + '/label:o')

        # Image and face
        self.width_img = rf.check('width', yarp.Value(320),
                                  'Width of the input image').asInt()

        self.height_img = rf.check('height', yarp.Value(244),
                                   'Height of the input image').asInt()

        self.face_coord_port.open('/' + self.module_name + '/coord:i')
        self.face_coord_port.setStrict(False)

        self.image_in_port.open('/' + self.module_name + '/image:i')
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()

        self.opc_port.open('/' + self.module_name + '/OPC:rpc')
        self.threshold_multimodal = 0.8

        info("Initialization complete")
        return success

    def load_model_audio(self):
        self.resample_trans = torchaudio.transforms.Resample(self.sampling_rate, 16000)

        # Load Database  for audio embeddings
        try:
            self.db_embeddings_audio = EmbeddingsHandler(os.path.join(self.dataset_path, "audio"), n_neighbors=4)
            self.model_audio = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        except FileNotFoundError:
            info(f"Unable to find dataset {EmbeddingsHandler(os.path.join(self.dataset_path, 'audio'))}")
            return False
        return True

    def load_model_face(self):
        try:
            self.modele_face = torch.load(self.face_model_path)
            self.modele_face.eval()

            self.db_embeddings_face = EmbeddingsHandler(os.path.join(self.dataset_path, "face"), threshold=self.threshold_face)

            # Transform for  face embeddings
            self.trans = transforms.Compose([
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization,
                transforms.Resize((180, 180))
            ])

        except FileNotFoundError:
            info(f"Unable to find dataset   {EmbeddingsHandler(os.path.join(self.dataset_path, 'face'))} \
            or model {self.face_model_path}")
            return False

        return True

    def interruptModule(self):
        print("[INFO] Stopping the module")
        self.audio_in_port.interrupt()
        self.label_outputPort.interrupt()
        self.eventPort.interrupt()
        self.handle_port.interrupt()
        self.image_in_port.interrupt()
        self.face_coord_port.interrupt()

        return True

    def close(self):
        self.audio_in_port.close()
        self.handle_port.close()
        self.label_outputPort.close()
        self.image_in_port.close()
        self.eventPort.close()
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

        elif command.get(0).asString() == "predict":
            self.predict = True
            reply.addString("ok")

        elif command.get(0).asString() == "stop":
            self.process = False
            reply.addString("ok")

        elif command.get(0).asString() == "predict":
            if command.get(1).asString() == "stop":
                self.predict = False
                reply.addString("ok")

        elif command.get(0).asString() == "check":
            if command.get(1).asString() == "tracker":
                new_detection = []
                new_detection.append(command.get(2).asList().get(0).asDouble())
                new_detection.append(command.get(2).asList().get(1).asDouble())
                new_detection.append(command.get(2).asList().get(2).asDouble())
                new_detection.append(command.get(2).asList().get(3).asDouble())

                name_to_assign, id_to_assign = self.check_existing_face(new_detection)
                if name_to_assign:
                    reply.addString(name_to_assign)
                    reply.addString(id_to_assign)
                else:
                    reply.addString("nack")

        elif command.get(0).asString() == "save":
            if command.get(1).asString() == "face":
                if command.get(2).asString() == "start":
                    self.save_face = True
                else:
                    name = command.get(2).asString().lower()
                    if name in self.db_embeddings_face.data_dict.keys():
                        self.db_embeddings_face.data_dict[name] = self.db_embeddings_face.data_dict[name] + self.face_emb
                    else:
                        self.db_embeddings_face.data_dict[name] = self.face_emb

                    self.database.save_faces(self.faces_img, self.face_emb, name)
                    self.save_face = False
                    self.faces_img = []
                    self.face_emb = []

            reply.addString("ok")

        elif command.get(0).asString() == "reset":
            self.db_embeddings_face.excluded_entities = []

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                if command.get(2).asString() == "audio":
                    self.threshold_audio = command.get(3).asDouble()
                    self.db_embeddings_audio.threshold = self.threshold_audio

                    reply.addString("ok")
                elif command.get(2).asString() == "face":
                    self.threshold_face = command.get(3).asDouble() if command.get(3).asDouble() > 0 else self.threshold_face
                    self.db_embeddings_face.threshold = self.threshold_face
                    reply.addString("ok")
                else:
                    reply.addString("nack")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                if command.get(2).asString() == "audio":
                    reply.addDouble(self.threshold_audio)
                elif command.get(2).asString() == "face":
                    reply.addDouble(self.threshold_face)
                else:
                    reply.addString("nack")
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

        if self.sound and self.is_voice:

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

    def check_voice(self):
        if self.eventPort.getInputCount():
            event_name = self.eventPort.read(False)
            if event_name:
                event_name = event_name.get(0).asString()
                if event_name == "start_voice":
                    self.is_voice = True
                elif event_name == "stop_voice":
                    self.audio = []
                    self.nb_samples_received = 0
                    self.is_voice = False
                else:
                    pass

    def get_face_coordinate(self):
        if self.face_coord_port.getInputCount():
            self.coord_face = self.face_coord_port.read(False)
            return self.coord_face is not None

        self.coord_face = None
        return False

    def set_name_memory(self, face_id, face_name):
        if self.opc_port.getOutputCount():
            reply = yarp.Bottle()
            cmd = yarp.Bottle("ask")
            list_condition = cmd.addList()
            cond1 = list_condition.addList()
            cond1.addString("id_tracker")
            cond1.addString("==")
            cond1.addString(face_id)

            self.opc_port.write(cmd, reply)
            list_id = reply.get(1).asList().get(1).asList()

            if list_id.size():
                cmd = yarp.Bottle()

                cmd.addString("get")
                list_all = cmd.addList()
                list_1 = list_all.addList()
                list_1.addString("id")
                list_1.addInt(list_id.get(0).asInt())

                list_2 = list_all.addList()
                list_2.addString("propSet")
                list_3 = list_2.addList()
                list_3.addString("verified")

                reply_ver = yarp.Bottle()

                self.opc_port.write(cmd, reply_ver)
                print("Sent cmd to OPC {}, and received response {}".format(cmd.toString(), reply_ver.toString()))

                verified = reply_ver.get(1).asList().get(0).asList().get(1).asInt()
                if verified == 0:
                    reply2 = yarp.Bottle()
                    cmd = yarp.Bottle()
                    cmd.addString("set")
                    list_cmd = cmd.addList()
                    id_cmd = list_cmd.addList()
                    id_cmd.addString("id")
                    id_cmd.addInt(list_id.get(0).asInt())
                    label_cmd = list_cmd.addList()
                    label_cmd.addString("label_tracker")
                    label_cmd.addString(face_name.strip())
                    # cmd_str = "set ((id " + str(list_id.get(0).asInt()) + ") (label_tracker" + face_name + "))"

                    self.opc_port.write(cmd, reply2)
                    print("Sent cmd to OPC {} and received reply {}".format(cmd.toString(), reply2.toString()))
                    return "ack" + reply2.get(0).asString()
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


            for i in range(list_id.size()):
                cmd_str = "get ((id " + str(list_id.get(i).asInt()) + ") (propSet (label_tracker)))"
                cmd = yarp.Bottle(cmd_str)
                reply_id = yarp.Bottle()
                self.opc_port.write(cmd, reply_id)
                if reply_id.size() > 0:
                    name = reply_id.get(1).asList().get(0).asList().get(1).asString()
                    self.db_embeddings_face.excluded_entities.append(name)
                    self.db_embeddings_audio.excluded_entities.append(name)

    def get_name_to_verify(self):

        if self.opc_port.getOutputCount():

            reply = yarp.Bottle()
            cmd = yarp.Bottle("ask")
            list_condition = cmd.addList()
            cond1 = list_condition.addList()
            cond1.addString("verified")
            cond1.addString("==")
            cond1.addInt(1)
            list_condition.addString("&&")
            cond2 = list_condition.addList()
            cond2.addString("active")
            cond2.addString("==")
            cond2.addInt(0)

            self.opc_port.write(cmd, reply)
            list_id = reply.get(1).asList().get(1).asList()

            name_to_verify = []
            id_to_verify = []
            if list_id.size() > 0:
                reply_id = yarp.Bottle()
                for i in range(list_id.size()):
                    cmd_str = "get ((id " + str(list_id.get(i).asInt()) + ") (propSet (label_tracker id_tracker)))"
                    cmd = yarp.Bottle(cmd_str)
                    self.opc_port.write(cmd, reply_id)
                    name = reply_id.get(1).asList().get(1).asList().get(1).asString()
                    id = reply_id.get(1).asList().get(0).asList().get(1).asString()
                    name_to_verify.append(name)
                    id_to_verify.append(id)

            return name_to_verify, id_to_verify

        return False

    def updateModule(self):
        current_face_emb = []
        current_id_faces = []
        speaker_name, audio_score = "unknown", 0

        self.check_voice()
        record_image = self.read_image()
        record_audio = self.record_audio()

        self.get_name_in_memory()
        self.get_face_coordinate()

        if self.process:

            if record_audio and self.nb_samples_received >= self.length_input * self.sound.getFrequency():
                    print("Computing Speaker Embedding")
                    audio_signal = self.format_signal(self.audio)
                    # Compute speaker embeddings and do speaker prediction only if the audio database is updated with
                    # the same people folders as the face embedding folders (make empty folders?)
                    self.speaker_emb = self.get_audio_embeddings(audio_signal)
                    self.audio = []
                    self.nb_samples_received = 0
                    speaker_name, audio_score = self.predict_speaker(self.speaker_emb)

            if record_image and self.frame.size != 0 and self.coord_face:
                try:
                    current_id_faces, self.coord_face = format_face_coord(self.coord_face)
                    face_img = [face_alignement(f, self.frame) for f in self.coord_face]
                    current_face_emb = self.get_face_embeddings(face_img)

                    if self.save_face and len(current_face_emb) > 0:

                        self.faces_img = self.faces_img + face_img
                        self.face_emb.append(current_face_emb[0].numpy())

                except Exception as e:
                    info("Exception while computing face embeddings" + str(e))

            if self.predict:
                if speaker_name != 'unknown' and len(current_face_emb):
                    info("Got Audio and Face embeddings")
                    faces_name, face_scores = self.predict_face(current_face_emb)

                    unknown_faces = []
                    distances = []
                    for face_id, emb, name, score in zip(current_id_faces, current_face_emb, faces_name, face_scores):
                        if name != "unknown":
                            name = self.format_name(name)
                            self.set_name_memory(face_id, name)
                            print("Predicted for face_id {} : {} with score {}".format(face_id, name, score))
                        else:
                            distances.append(self.db_embeddings_face.get_distance_from_user(emb, speaker_name))
                            unknown_faces.append(face_id)

                    if len(unknown_faces):
                        min_distance_index = np.argmax(distances)
                        min_face_id = unknown_faces.pop(min_distance_index)
                        self.set_name_memory(min_face_id, speaker_name)
                        # print("Speaker name closest to unknown face is {} ".format(speaker_name))

                        for face_id in unknown_faces:
                            self.set_name_memory(face_id, "unknown")

                elif len(current_face_emb):
                    faces_name, scores = self.predict_face(current_face_emb)
                    for face_id, name, score in zip(current_id_faces, faces_name, scores):
                        self.set_name_memory(face_id, name)
                        print("Predicted for face_id {} : {} with score {}".format(face_id, name, score))
                else:
                    pass
        return True

    def check_existing_face(self, detection):
        users_to_verify, id_to_verify = self.get_name_to_verify()
        face_name = ""
        face_id = ""
        if len(users_to_verify) > 0:
            face_img_list = []
            face_img = face_alignement(detection, self.frame)
            face_img_list.append(face_img)
            current_face_emb = self.get_face_embeddings(face_img_list)

            if len(current_face_emb):
                distances = []
                names = []
                ids = []

                current_face_emb = current_face_emb[0]
                for (user, id) in zip(users_to_verify, id_to_verify):
                    # if user exist in db_embedding folder
                    distances.append(self.db_embeddings_face.get_distance_from_user(current_face_emb, user))
                    names.append(user)
                    ids.append(id)

                # max similarity is min distance (cosine similarity output [-1,1]
                min_distance_index = np.argmax(distances)
                face_name = names[min_distance_index]
                face_id = ids[min_distance_index]

        return face_name, face_id

    def format_signal(self, audio_list_samples):
        """
        Format an audio given a list of samples
        :param audio_list_samples:
        :return: numpy array
        """
        np_audio = np.concatenate(audio_list_samples, axis=1)
        np_audio = np.squeeze(np_audio)
        signal = np.transpose(np_audio, (1, 0))
        signal = signal.mean(axis=1)

        return signal

    def get_audio_embeddings(self, audio):
        """
        Generate voice embedding from audio sample
        :param audio:
        :return:
        """
        resample_audio = self.resample_trans(torch.from_numpy(audio.transpose()))
        embedding = self.model_audio.encode_batch(resample_audio)
        embedding = embedding.squeeze(axis=0)

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

        if score == -1:
            speaker_name = "unknown"

        self.db_embeddings_audio.excluded_entities = []
        print("Predicted speaker name is {} with score {}".format(speaker_name, score))

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
        self.db_embeddings_face.excluded_entities = []
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

    def format_name(self, name):
        name.strip()
        return name


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
