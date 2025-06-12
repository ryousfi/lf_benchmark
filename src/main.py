from comet import download_model, load_from_checkpoint
import sacrebleu
import json

def calculate_scores(hypothesis_data_file_path, reference_data_file_path, source_data_file_path, output_file_path):

    # and calculate the BLUE and COMET scores
    with open(hypothesis_data_file_path, "r", encoding="utf-8") as f1, \
         open(reference_data_file_path, "r", encoding="utf-8") as f2, \
         open(source_data_file_path, "r", encoding="utf-8") as f3:
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
        with open(output_file_path, "w", encoding="utf-8") as output_file:
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
    # ask the user to select one or more language pairs of the list
    print("Available language pairs:")
    print("1. EN-DE (English to German)")
    print("2. EN-FR (English to French)")
    print("3. EN-IT (English to Italian)")
    print("Please select a language pair by entering the corresponding number (1-3) or type 'all' to process all pairs.")
    # get the user input
    user_input = input("Enter your choice: ").strip().lower()
    if user_input == 'all':
        print("Processing all language pairs...")
    else:
        try:
            choice = int(user_input)
            if choice < 1 or choice > 3:
                print("Invalid choice. Please enter a number between 1 and 3 or type 'all'.")
                return
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 3 or type 'all'.")
            return

    en_de_executions = [
        {
            "hypothesis_data_file_path": "data/output/lf_0/LF-0-output-DE-CH.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_0_EN_DE-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_a/LF-a-output-DE-CH.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_a_EN_DE-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_b/LF-b-output-DE-CH.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_b_EN_DE-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_c/LF-c-output-DE-CH.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_c_EN_DE-CH.csv"
        }
    ]

    en_fr_executions = [
        {
            "hypothesis_data_file_path": "data/output/lf_0/LF-0-output-FR-CH.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_0_EN_FR-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_a/LF-a-output-FR-CH.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_a_EN_FR-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_b/LF-b-output-FR-CH.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_b_EN_FR-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_c/LF-c-output-FR-CH.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_c_EN_FR-CH.csv"
        }
    ]

    en_it_executions = [
        {
            "hypothesis_data_file_path": "data/output/lf_0/LF-0-output-IT-CH.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_0_EN_IT-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_a/LF-a-output-IT-CH.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_a_EN_IT-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_b/LF-b-output-IT-CH.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_b_EN_IT-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_c/LF-c-output-IT-CH.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_c_EN_IT-CH.csv"
        }
    ]

    # Execution 2


    en_de_executions2 = [
        {
            "hypothesis_data_file_path": "data/output/lf_0/LF-0-output-DE-DE.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_0_EN_DE-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_a/LF-a-output-DE-DE.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_a_EN_DE-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_b/LF-b-output-DE-DE.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_b_EN_DE-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_c/LF-c-output-DE-DE.json",
            "reference_data_file_path": "data/input/ref_de-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_c_EN_DE-CH.csv"
        }
    ]

    en_fr_executions2 = [
        {
            "hypothesis_data_file_path": "data/output/lf_0/LF-0-output-FR-FR.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_0_EN_FR-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_a/LF-a-output-FR-FR.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_a_EN_FR-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_b/LF-b-output-FR-FR.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_b_EN_FR-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_c/LF-c-output-FR-FR.json",
            "reference_data_file_path": "data/input/ref_fr-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_c_EN_FR-CH.csv"
        }
    ]

    en_it_executions2 = [
        {
            "hypothesis_data_file_path": "data/output/lf_0/LF-0-output-IT-IT.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_0_EN_IT-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_a/LF-a-output-IT-IT.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_a_EN_IT-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_b/LF-b-output-IT-IT.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_b_EN_IT-CH.csv"
        },
        {
            "hypothesis_data_file_path": "data/output/lf_c/LF-c-output-IT-IT.json",
            "reference_data_file_path": "data/input/ref_it-CH.json",
            "source_data_file_path": "data/input/src_en-CH.json",
            "output_file_path": "dist/scores_lf_c_EN_IT-CH.csv"
        }
    ]

    


    if user_input == 'all':
        print("Processing EN-DE, EN-FR, and EN-IT language pairs...")
        # Combine all executions into a single list
        executions = []
        executions.extend(en_de_executions2)
        executions.extend(en_fr_executions2)
        executions.extend(en_it_executions2)

    elif choice == 1:
        print("Processing EN-DE language pair...")
        executions = en_de_executions
    elif choice == 2:
        print("Processing EN-FR language pair...")
        executions = en_fr_executions
    elif choice == 3:
        print("Processing EN-IT language pair...")
        executions = en_it_executions

    for execution in executions:
        calculate_scores(
            execution["hypothesis_data_file_path"],
            execution["reference_data_file_path"],
            execution["source_data_file_path"],
            execution["output_file_path"]
        )


if __name__ == "__main__":
    main()