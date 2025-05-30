import click
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tensorflow.keras.models import load_model
from tqdm import tqdm
from typing import Any
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

def get_protT5_features(
    sequence: str,
    tokenizer: T5Tokenizer,
    pretrained_model: T5EncoderModel,
    device: torch.device
) -> np.ndarray:
    sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = [" ".join(sequence)]
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)
    with torch.no_grad():
        embedding = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()
    seq_len = int((attention_mask[0] == 1).sum())
    seq_emd = embedding[0][: seq_len - 1]
    return seq_emd

@click.command()
@click.option(
    '--input-fasta-file', '-i',
    required=True,
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help='Input FASTA file path'
)
@click.option(
    '--output-csv-file', '-o',
    required=True,
    type=click.Path(writable=True, dir_okay=False),
    help='Output CSV file path'
)
@click.option(
    '--model-path', '-m',
    default="models/NGlyDE_Prot_T5_Final.h5",
    show_default=True,
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help='Keras model file path'
)
def main(input_fasta_file: str, output_csv_file: str, model_path: str) -> None:
    cutoff_threshold: float = 0.5

    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )
    pretrained_model: T5EncoderModel = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()

    device: torch.device = torch.device("cpu")
    pretrained_model = pretrained_model.to(device)
    pretrained_model = pretrained_model.eval()

    results_df: pd.DataFrame = pd.DataFrame(
        columns=["prot_desc", "position", "site_residue", "probability", "prediction"]
    )

    for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
        prot_id: str = seq_record.id
        sequence: str = str(seq_record.seq)
        pt5_all: np.ndarray = get_protT5_features(sequence, tokenizer, pretrained_model, device)
        for index, amino_acid in enumerate(sequence):
            if amino_acid == "N":
                site: int = index + 1
                X_test_pt5: np.ndarray = pt5_all[index]
                combined_model: Any = load_model(model_path)
                y_pred: float = combined_model.predict(
                    np.array(X_test_pt5.reshape(1, 1024)), verbose=0
                )[0][0]
                results_df.loc[len(results_df)] = [
                    prot_id,
                    site,
                    amino_acid,
                    1 - y_pred,
                    int(y_pred < cutoff_threshold),
                ]

    print("Saving results ...")
    results_df.to_csv(output_csv_file, index=False)
    print("Results saved to " + output_csv_file)

if __name__ == "__main__":
    main()
