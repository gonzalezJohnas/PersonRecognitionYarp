import argparse
from project.utils import save_audio, split_audio_chunks, process_dataset, save_gammatone
import os, glob


def main(args, ):
    # Create the output dir
    output_dir = os.path.join(args.output_dir)
    os.mkdir(output_dir)
    dirs = os.listdir(args.root_dir)

    for d in dirs:
        audio_files = glob.glob(os.path.join(args.root_dir, d, "*.wav"))

        os.mkdir(os.path.join(output_dir, d))
        chunks = []
        fs = 0
        for f in audio_files:
            fs, chunk_c1, chunk_c2 = split_audio_chunks(f, args.length, args.overlap)
            chunks += chunk_c1 + chunk_c2

        if args.saving_type == "gammatone":
            save_gammatone(chunks, os.path.join(output_dir, d), fs)
        else:
            save_audio(chunks, os.path.join(output_dir, d), fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset of speaker, split each audio into chunk")
    parser.add_argument("root_dir", type=str, help="Path to the dataset")
    parser.add_argument("length", type=int, help="Length of chunks in milliseconds")
    parser.add_argument("overlap", type=int, help="Overlap between  chunks in milliseconds")
    parser.add_argument("output_dir", type=str, help="Output dir")
    parser.add_argument("--saving_type", type=str, help="Saving raw audio (raw) or gammatone (gammatone)",
                        default="save_audio")
    args = parser.parse_args()

    main(args)