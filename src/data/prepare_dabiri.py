import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def prepare_dabiri(docx_path="data/raw/41598_2024_54010_MOESM1_ESM.docx", output_path="data/dabiri_2024.csv"):
    """Extracts Table S1 (565 wellhead choke tests) from the Dabiri et al. (2024) supplementary file."""
    root = ET.fromstring(zipfile.ZipFile(docx_path).read("word/document.xml"))
    table = root.find(W + "body").find(W + "tbl")  # Table S1 is the first table

    rows = []
    for tr in table.findall(W + "tr")[1:]:  # skip header row
        cells = ["".join(t.text or "" for t in tc.iter(W + "t")).strip() for tc in tr.findall(W + "tc")]
        rows.append(cells)

    # Header in the source reads 'GLR(SCF/D)'; the paper defines GLR in SCF/STB
    df = pd.DataFrame(rows, columns=['No', 'Pwh_psi', 'GLR_scf_stb', 'Choke_64ths', 'Q_liquid_STB']).astype(float)
    df['No'] = df['No'].astype(int)

    df.to_csv(output_path, index=False)
    print(f"Extracted {len(df)} records ({df.duplicated(subset=df.columns[1:]).sum()} exact duplicates) to '{output_path}'.")
    return df

if __name__ == "__main__":
    prepare_dabiri()
