from comet import download_model, load_from_checkpoint
import sacrebleu
                                                                              

def calculate_blue_score():
    # read hyp.txt and ref.txt line by line
    print("BLUE scores ************************")
    with open("data/hyp.txt", "r") as f1, open("data/ref.txt", "r") as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
        # for each line in file 1, calculate the score
        i = 0
        total = 0
        for line1, line2 in zip(file1_lines, file2_lines):
            # remove new line characters
            line1 = line1.strip()
            line2 = line2.strip()

            bleu = sacrebleu.sentence_bleu(
                line1,
                [line2],
                lowercase=True,
                tokenize='none',
                use_effective_order=True
            )
            total = total + bleu.score
            print(f"Score {i}: {bleu.score:.1f}")
            i = i + 1
        print(f"Overall score: {total/i+1:.1f}")


# a method returning value
def calculate_comet_score():
    # Choose your model from Hugging Face Hub
    # model_path = download_model("Unbabel/XCOMET-XL")
    # or for example:
    model_path = download_model("Unbabel/wmt22-comet-da")
    print(f"PATH: {model_path}")
    # Load the model checkpoint:
    model = load_from_checkpoint(model_path)
    # Data must be in the following format:

    data = []

    i = 0
    total = 0
    # read src.txt, ref.txt, hyp.txt line by line
    with open("data/src.txt", "r") as f1, open("data/ref.txt", "r") as f2, open("data/hyp.txt", "r") as f3:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
        file3_lines = f3.readlines()
        
        for line1, line2, line3 in zip(file1_lines, file2_lines, file3_lines):
            # remove new line characters
            line1 = line1.strip()
            line2 = line2.strip()
            line3 = line3.strip()
            
            data.append(
                {
                    "src": line1,
                    "mt": line2,
                    "ref": line3
                }
            )


    # Call predict method:
    # loop through the data and get the index of the data
    model_output = model.predict(data, batch_size=8, gpus=1)
    # print the model output in  the following format
    # Score 0: 0.1234
    # Score 1: 0.5678
    # Score 2: 0.9101
    # Score 3: 0.1121
    # Score 4: 0.3141
    # Score 5: 0.5161
    print("COMET scores ************************")
    for i, score in enumerate(model_output.scores):
        print(f"Score {i}: {score*100:.1f}")

    # print the overall score in the following format
    # Overall score: 0.1234
    print(f"Overall score: {model_output.system_score*100:.1f}")


def main():
    calculate_comet_score()
    calculate_blue_score()


if __name__ == "__main__":
    main()