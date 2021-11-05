from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch



checkpoint_path = "./training_saved_models/trunk_best150.pth"
model_save_path = "./saved_model/model_triple_facerecogntion_150.pt"

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare the model
    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2",
        dropout_prob=0.7
    ).to(device)

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint)
    torch.save(model, model_save_path)