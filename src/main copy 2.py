from comet import download_model, load_from_checkpoint
import sacrebleu
import json



def read_json_files(hypothesis_data_file_path, reference_data_file_path, source_data_file_path, output_file_path):

    # and calculate the BLUE and COMET scores
    with open(hypothesis_data_file_path, "r") as f1, open(reference_data_file_path, "r") as f2, open(source_data_file_path, "r") as f3:
        hyp_data = json.load(f1)
        ref_data = json.load(f2)
        src_data = json.load(f3)

        i = 0
        blue_scores = []
        comet_scores = []
        comet_data = []

        for hyp_entry, ref_entry, src_entry in zip(hyp_data, ref_data, src_data):
            # print the value of each entry
            hyp_value = hyp_data[hyp_entry]
            ref_value = ref_data[ref_entry]
            src_value = src_data[src_entry]

            # BLUE score
            blue_score_value = calculate_bleu(hyp_value, ref_value)
            blue_scores.append(blue_score_value)
            print(f"Blue score {i}: {blue_score_value:.1f}")


            # COMET score
            comet_data.append(
                {
                    "src": src_value,
                    "mt": hyp_value,
                    "ref": ref_value    
                }
            )

            i = i + 1


        comet_score = calculate_comet(comet_data)
        for i, score in enumerate(comet_score.scores):
            comet_scores.append(score*100)
        

        # write blue_scores into a csv file named output.csv
        with open(output_file_path, "w") as output_file:
            output_file.write("blue_score,comet_score\n")
            for blue, comet in zip(blue_scores, comet_scores):
                output_file.write(f"{blue},{comet}\n")


# a method that takes two parameters and return the sum of the two parameters
def calculate_bleu(hyp, ref):
    bleu = sacrebleu.sentence_bleu(
                hyp,
                [ref],
                lowercase=True,
                tokenize='none',
                use_effective_order=True
            )
    return bleu.score

def calculate_comet(data):
    model_path = download_model("Unbabel/wmt22-comet-da")
    # Load the model checkpoint:
    model = load_from_checkpoint(model_path)
    score = model.predict(data, batch_size=8, gpus=1)
    return score



def main():
    hypothesis_data_file_path = "data/hyp.json"
    reference_data_file_path = "data/ref.json"
    source_data_file_path = "data/src.json"
    output_file_path = "dist/output.csv"
    read_json_files(hypothesis_data_file_path, reference_data_file_path, source_data_file_path, output_file_path)



if __name__ == "__main__":
    main()