import os
import anndata
from typing import Dict, List, Optional
import re
import warnings

# Suppress annoying anndata/scanpy warnings during file loading
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. DEFINING THE MAPPING (Keys are corrected to match the actual files on disk, especially those with ..h5ad) ---
from typing import Dict

FULL_CLINICAL_MAPPING: Dict[str, str] = {
    # Note: These map raw fragment files (*_fragments.tsv.gz) to processed AnnData files (*_ATAC.h5ad)
    'GSM7789963_Patient_5_35A4AL_ATAC.h5ad': 'B_B5_Breast_ERneg_Unknown_ATAC.h5ad',
    'GSM7789965_Patient_7_35EE8L_ATAC.h5ad': 'B_B7_Breast_TNBC_Unknown_ATAC.h5ad',
    'GSM7789967_Patient_8_3821AL_ATAC.h5ad': 'B_B8_Breast_ERpos_Equivocal_ATAC.h5ad',
    'GSM7789969_Patient_9_3B3E9L_ATAC.h5ad': 'B_B9_Breast_ERpos_Equivocal_ATAC.h5ad',
    'GSM7789971_Patient_10_3C7D1L_ATAC.h5ad': 'B_B10_Breast_ERpos_Equivocal_ATAC.h5ad',
    'GSM7789973_Patient_13_3D388L_ATAC.h5ad': 'B_B13_Breast_ERpos_Her2neg_ATAC.h5ad',
    'GSM7789975_Patient_11_3FCDEL_ATAC.h5ad': 'B_B11_Breast_ERpos_Her2neg_ATAC.h5ad',
    'GSM7789977_Patient_14_43E7BL_ATAC.h5ad': 'B_B14_Breast_ERpos_Her2neg_ATAC.h5ad',
    'GSM7789979_Patient_14_43E7CL_ATAC.h5ad': 'B_B14B_Breast_ERpos_Her2neg_ATAC.h5ad',
    'GSM7789981_Patient_12_44F0AL_ATAC.h5ad': 'B_B12_Breast_ERpos_Equivocal_ATAC.h5ad',
    'GSM7789983_Patient_15_45CB0L_ATAC.h5ad': 'B_B15_Breast_ERpos_Equivocal_ATAC.h5ad',
    'GSM7789993_Patient_6_4C2E5L_ATAC.h5ad': 'B_B6_Breast_TNBC_Unknown_ATAC.h5ad',
    'GSM7789985_Patient_1_49758L_ATAC.h5ad': 'B_B1_NormalBreast_NonCancer_Sensitive_ATAC.h5ad',
    'GSM7789987_Patient_2_49CFCL_ATAC.h5ad': 'B_B2_NormalBreast_NonCancer_Sensitive_ATAC.h5ad',
    'GSM7789989_Patient_3_4AF75L_ATAC.h5ad': 'B_B3_NormalBreast_NonCancer_Sensitive_ATAC.h5ad',
    'GSM7789991_Patient_4_4B146L_ATAC.h5ad': 'B_B4_NormalBreast_NonCancer_Sensitive_ATAC.h5ad',
    'GSM7789996_HCC1143_ATAC.h5ad': 'CellLine_HCC1143_Breast_TNBC_Sensitive_ATAC.h5ad',
    'GSM7789997_MCF7_ATAC.h5ad': 'CellLine_MCF7_Breast_ERpos_Chemonaive_ATAC.h5ad',
    'GSM7790000_SUM149PT_ATAC.h5ad': 'CellLine_SUM149PT_Breast_TNBC_Resistant_ATAC.h5ad',
    'GSM7790001_T47D_ATAC.h5ad': 'CellLine_T47D_Breast_ERpos_Chemonaive_ATAC.h5ad',
    
    # Additional 'old names' to 'new names' (Mapping based on file names)
    'GSM5017686_MM468_5FU3_day202.h5ad': 'CellLine_MM468-5FU3-D202_TNBC_InVitro_TNBC.h5ad', # Highly speculative, no direct match
    'GSM5017687_MM468_5FU3_day50.h5ad': 'CellLine_MM468-5FU3-D50_TNBC_InVitro_TNBC.h5ad', # Day 50 (old) -> Day 33 (new) - Speculative
    'GSM5017688_MM468_5FU3_day77.h5ad': 'CellLine_MM468-5FU3-D77_TNBC_InVitro_TNBC.h5ad',
    'GSM5017689_MM468_5FU2_day171.h5ad': 'CellLine_MM468-5FU2-D171_TNBC_InVitro_TNBC.h5ad',
    'GSM5017690_MM468_5FU2_day67.h5ad': 'CellLine_MM468-5FU2-D67_TNBC_InVitro_TNBC.h5ad',
    'GSM5017691_MM468_5FU1_day214.h5ad': 'CellLine_MM468-5FU1-D214_TNBC_InVitro_TNBC.h5ad', # Day 214 (old) -> Day 171 (new) - Speculative
    'GSM5017692_MM468_5FU1_day33.h5ad': 'CellLine_MM468-5FU1-D33_TNBC_InVitro_TNBC.h5ad',
    'GSM5017693_MM468_5FU1_UNC_day33.h5ad': 'CellLine_MM468-5FUUnt-D33-H3K4me3_TNBC_InVitro_TNBC.h5ad', # UNC_day33 (old) -> Untreated-D0 (new) - Speculative
    'GSM5017694_MM468_chemonaive.h5ad': 'CellLine_MM468-Untreated-D0-H3K4me3_TNBC_InVitro_TNBC.h5ad',
    'GSM5017695_MM468_UNC_day33.h5ad': 'CellLine_MM468-Untreated-D33-H3K4me3_TNBC_InVitro_TNBC.h5ad',
    'GSM5017696_HBCx95_chemonaive.h5ad': 'C_C1_Breast_Unknown_Chemonaive.h5ad', # HBCx95 -> PDX95
    'GSM5017697_HBCx95_persister.h5ad': 'C_C1_Breast_Unknown_Persister.h5ad', # HBCx95 -> PDX95
    'GSM5017698_HBCx95_recurrent.h5ad': 'C_C1_Breast_Unknown_Recurrent.h5ad', # Recurrent to Recurrent (same as chemonaive) - Speculative
    'GSM5017699_HBCx95_residual.h5ad': 'C_C1_Breast_Unknown_Residual.h5ad', # Residual to Residual (same as persister) - Speculative
    'GSM5017700_HBCx95_resistant.h5ad': 'C_C1_Breast_Unknown_Resistant.h5ad', # Resistant to Recurrent - Speculative
    'GSM5662564_HBCx39_chemonaive.h5ad': 'C_C2_Breast_Unknown_Chemonaive.h5ad', # HBCx39 -> SUM149PT - Highly speculative
    'GSM5662565_HBCx39_persister.h5ad': 'C_C2_Breast_Unknown_Persister.h5ad', # HBCx39 -> HCC1143 - Highly speculative
    'GSM5662566_HBCx172_chemonaive.h5ad': 'C_C3_Breast_Unknown_Chemonaive.h5ad', # HBCx172 -> MCF7 - Highly speculative
    'GSM5662567_HBCx172_persister.h5ad': 'C_C3_Breast_Unknown_Persister.h5ad', # HBCx172 -> T47D - Highly speculative
    
    'GSM5276933-3533EL.h5ad': 'A_A1_Endometrial_IA_Unknown.h5ad',
    'GSM5276934-3571DL.h5ad': 'A_A2_Endometrial_IA_Unknown.h5ad',
    'GSM5276935-36186L.h5ad': 'A_A3_Endometrial_IA_Unknown.h5ad',
    'GSM5276936-36639L.h5ad': 'A_A4_Endometrial_IA_Unknown.h5ad',
    'GSM5276937-366C5L.h5ad': 'A_A5_Endometrial_IA_Unknown.h5ad',
    'GSM5276938-37EACL.h5ad': 'A_A6_Endometrial_IIIA_Unknown.h5ad',
    'GSM5276939-38FE7L.h5ad': 'A_A7_Ovarian_IA_Unknown.h5ad',
    'GSM5276940-3BAE2L.h5ad': 'A_A8_Ovarian_IIB_Unknown.h5ad',
    'GSM5276941-3CCF1L.h5ad': 'A_A10_Ovarian_IVB_Unknown.h5ad',
    'GSM5276942-3E4D1L.h5ad': 'A_A11_Gastric_IV_Unknown.h5ad',
    'GSM5276943-3E5CFL.h5ad': 'A_A9_Ovarian_IIIC_Unknown.h5ad',
    
    # --- Patient Tumor Samples (Matched by Patient Number) ---
    'GSM7789964_Patient_5_35A4AL_RNA.h5ad': 'B_B5_Breast_ERneg_Unknown.h5ad',
    'GSM7789966_Patient_7_35EE8L_RNA.h5ad': 'B_B7_Breast_TNBC_Unknown.h5ad',
    'GSM7789968_Patient_8_3821AL_RNA.h5ad': 'B_B8_Breast_ERpos_Equivocal.h5ad',
    'GSM7789970_Patient_9_3B3E9L_RNA.h5ad': 'B_B9_Breast_ERpos_Equivocal.h5ad',
    'GSM7789972_Patient_10_3C7D1L_RNA.h5ad': 'B_B10_Breast_ERpos_Equivocal.h5ad',
    'GSM7789974_Patient_13_3D388L_RNA.h5ad': 'B_B13_Breast_ERpos_Her2neg.h5ad',
    'GSM7789976_Patient_11_3FCDEL_RNA.h5ad': 'B_B11_Breast_ERpos_Her2neg.h5ad',
    'GSM7789978_Patient_14_43E7BL_RNA.h5ad': 'B_B14_Breast_ERpos_Her2neg.h5ad',
    'GSM7789980_Patient_14_43E7CL_RNA.h5ad': 'B_B14B_Breast_ERpos_Her2neg.h5ad',
    'GSM7789982_Patient_12_44F0AL_RNA.h5ad': 'B_B12_Breast_ERpos_Equivocal.h5ad',
    'GSM7789984_Patient_15_45CB0L_RNA.h5ad': 'B_B15_Breast_ERpos_Equivocal.h5ad',
    'GSM7789994_Patient_6_4C2E5L_RNA.h5ad': 'B_B6_Breast_TNBC_Unknown.h5ad',
    'GSM7789986_Patient_1_49758L_RNA.h5ad': 'B_B1_NormalBreast_NonCancer_Sensitive.h5ad',
    'GSM7789988_Patient_2_49CFCL_RNA.h5ad': 'B_B2_NormalBreast_NonCancer_Sensitive.h5ad',
    'GSM7789990_Patient_3_4AF75L_RNA.h5ad': 'B_B3_NormalBreast_NonCancer_Sensitive.h5ad',
    'GSM7789992_Patient_4_4B146L_RNA.h5ad': 'B_B4_NormalBreast_NonCancer_Sensitive.h5ad',
    'GSM7789995_HCC1143_RNA.h5ad': 'CellLine_HCC1143_Breast_TNBC_Sensitive.h5ad',
    'GSM7789998_MCF7_RNA.h5ad': 'CellLine_MCF7_Breast_ERpos_Chemonaive.h5ad',
    'GSM7789999_SUM149PT_RNA.h5ad': 'CellLine_SUM149PT_Breast_TNBC_Resistant.h5ad',
    'GSM7790002_T47D_RNA.h5ad': 'CellLine_T47D_Breast_ERpos_Chemonaive.h5ad',
    

    
    'GSM5276944_3533EL_ATAC.h5ad': 'A_A1_Endometrial_IA_Unknown_ATAC.h5ad',
    'GSM5276945_3571DL_ATAC.h5ad': 'A_A2_Endometrial_IA_Unknown_ATAC.h5ad',
    'GSM5276946_36186L_ATAC.h5ad': 'A_A3_Endometrial_IA_Unknown_ATAC.h5ad',
    'GSM5276947_36639L_ATAC.h5ad': 'A_A4_Endometrial_IA_Unknown_ATAC.h5ad',
    'GSM5276948_366C5L_ATAC.h5ad': 'A_A5_Endometrial_IA_Unknown_ATAC.h5ad',
    'GSM5276949_37EACL_ATAC.h5ad': 'A_A6_Endometrial_IIIA_Unknown_ATAC.h5ad',
    'GSM5276950_38FE7L_ATAC.h5ad': 'A_A7_Ovarian_IA_Unknown_ATAC.h5ad',
    'GSM5276951_3BAE2L_ATAC.h5ad': 'A_A8_Ovarian_IIB_Unknown_ATAC.h5ad',
    'GSM5276952_3E5CFL_ATAC.h5ad': 'A_A9_Ovarian_IIIC_Unknown_ATAC.h5ad',
    'GSM5276953_3CCF1L_ATAC.h5ad': 'A_A10_Ovarian_IVB_Unknown_ATAC.h5ad',
    'GSM5276954_3E4D1L_ATAC.h5ad': 'A_A11_Gastric_IV_Unknown_ATAC.h5ad',
    
    
    # --- MM468 Cell Line Samples (DMSO/Untreated) ---
    'GSM5008855_MM468_DMSO1_day60_H3K27me3..h5ad': 'CellLine_MM468-Untreated-D60_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008856_MM468_DMSO3_day77_H3K27me3..h5ad': 'CellLine_MM468-Untreated-D77_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008857_MM468_DMSO5_day131_H3K27me3..h5ad': 'CellLine_MM468-Untreated-D131_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008863_MM468_DMSO1_day0_H3K4me3..h5ad': 'CellLine_MM468-Untreated-D0_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008858_MM468_5FU1_day33_H3K27me3..h5ad': 'CellLine_MM468-5FU1-D33_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008859_MM468_5FU2_day67_H3K27me3..h5ad': 'CellLine_MM468-5FU2-D67_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008860_MM468_5FU6_day131_H3K27me3..h5ad': 'CellLine_MM468-5FU6-D131_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008861_MM468_5FU3_day147_H3K27me3..h5ad': 'CellLine_MM468-5FU3-D147_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008862_MM468_5FU2_day171_H3K27me3..h5ad': 'CellLine_MM468-5FU2-D171_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5008864_MM468_5FU1_day60_H3K4me3..h5ad': 'CellLine_MM468-5FU1-D60-H3K4me3_TNBC_InVitro_TNBC_ATAC.h5ad',
    'GSM5782059_MM468_GSKJ4_day91_H3K27me3..h5ad': 'CellLine_MM468-GSK4-D91_TNBC_InVitro_TNBC_ATAC.h5ad', # GSKJ4 maps to GSK4
    'GSM5008865_HBCx95_m43_UNT_H3K27me3..h5ad': 'C_C1K27_Breast_Unknown_Untreated_ATAC.h5ad',
    'GSM5008866_HBCx95_m43_UNT_H3K4me3..h5ad': 'C_C1K4_PDX95_Unknown_Untreated_ATAC.h5ad',
    # round 2
    # 'PROCESSED_C_C1_Breast_Unknown_Untreated_H3K27me3_ATAC_singlets.h5ad':'PROCESSED_C_C1K27_Breast_Unknown_Untreated_H3K27me3_ATAC_singlets.h5ad',
    # 'PROCESSED_C_C1_PDX95_Unknown_Untreated_H3K4me3_ATAC_singlets.h5ad':'PROCESSED_C_C1K4_PDX95_Unknown_Untreated_H3K4me3_ATAC_singlets.h5ad',
}

def rename_files(directory: str, mapping: Dict[str, str]) -> List[str]:
    """
    Renames files in the specified directory based on the mapping dictionary.

    Args:
        directory: The path to the directory containing the files.
        mapping: A dictionary where keys are original filenames and values are new filenames.

    Returns:
        A list of the new filenames that should now exist in the directory.
    """
    print(f"--- 1. Starting file renaming in directory: {directory} ({len(mapping)} files) ---")
    files_to_process: List[str] = []

    for old_name, new_name in mapping.items():
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                # print(f"Renamed: {old_name} -> {new_name}")
                files_to_process.append(new_name)
            except OSError as e:
                print(f"ERROR renaming file {old_name}: {e}. Skipping.")
        else:
            # Check for both the old name and the new name, as some files might already be renamed
            if os.path.exists(new_path):
                files_to_process.append(new_name)
            else:
                print(f"WARNING: File not found: {old_name} (and {new_name}). Skipping rename. Check if the file is in the current directory.")

    print("--- File renaming complete. ---")
    return list(set(files_to_process)) # Return unique list of files now present in the new naming format

def process_and_add_patient_label(
    file_path: str,
    output_dir: str = ""
) -> Optional[anndata.AnnData]:
    """
    Loads an h5ad file, extracts patient/sample metadata from the filename,
    adds it to the AnnData object, and returns the modified object.

    Args:
        file_path: The full path to the h5ad file (using the NEW name).
        output_dir: Directory to save the processed file. If empty, the file
                    will be saved in the same directory.

    Returns:
        The processed AnnData object, or None if an error occurred.
    """
    print(f"\n--- 2. Processing file: {os.path.basename(file_path)} ---")
    
    # 1. Load AnnData
    if not os.path.exists(file_path):
        print(f"    ERROR: File not found: {os.path.basename(file_path)}. Skipping. **Please ensure this file is present in the directory to be processed.**")
        return None

    try:
        adata = anndata.read_h5ad(file_path)
    except Exception as e:
        print(f"    ERROR loading AnnData file {os.path.basename(file_path)}: {e}. Skipping.")
        return None

    new_filename = os.path.basename(file_path).replace(".h5ad", "")
    
    # 2. Extract Labels from the new filename (e.g., 'B_B2_NormalBreast_NonCancer_Sensitive')
    parts = new_filename.split('_')
     
    sample_prefix = parts[1] if len(parts) > 1 else parts[0]
    adata.obs.index = sample_prefix + '_' + adata.obs.index.astype(str)

    # 3. Apply labels to the AnnData object
    # The cell barcode is the index, we want to add a column to obs.
    adata.obs['Batch_ID'] = parts[0]
    adata.obs['Sample_ID'] = parts[1]
    adata.obs['Cancer_Type'] = parts[2]
    adata.obs['Cancer_Status'] = parts[3]
    adata.obs['Response_Status'] = parts[4]
    
    # 4. Save the modified AnnData object
    output_path = os.path.join(output_dir if output_dir else os.path.dirname(file_path), f"PROCESSED_{new_filename}.h5ad")

    try:
        adata.write(output_path)
        print(f"    SUCCESS: Processed file saved to {output_path}")
        return adata
    except Exception as e:
        print(f"    ERROR saving AnnData file {os.path.basename(file_path)}: {e}. Skipping.")
        return None

def main():
    """
    Main function to run the file renaming and processing pipeline.
    """
    # Directory where the script is executed. Assumed to be the data directory.
    # The script execution command: python3 /path/to/script.py
    # The current directory is: /mnt/f/H3K27/data
    data_directory = os.path.abspath(os.getcwd())
    
    # 1. Rename files
    renamed_files: List[str] = rename_files(data_directory, FULL_CLINICAL_MAPPING)

    if not renamed_files:
        print("\n--- 3. AnnData preparation failed. No files were loaded successfully. ---")
        return
        
    loaded_anndatas: List[anndata.AnnData] = []
    
    # 2. Process each newly named file
    for new_filename in renamed_files:
        file_path = os.path.join(data_directory, new_filename)
        
        # We process files that now exist under the *new* name
        processed_adata = process_and_add_patient_label(file_path)
        
        if processed_adata is not None:
            loaded_anndatas.append(processed_adata)
            
    if not loaded_anndatas:
        print("\n--- 3. AnnData preparation failed. No files were loaded successfully. ---")
    else:
        # Optional: You can add code here to concatenate all loaded_anndatas if needed
        print(f"\n--- 3. AnnData preparation complete. {len(loaded_anndatas)} files successfully processed and labeled. ---")
        # Example of saving a concatenated object (uncomment if desired):
        # concatenated_adata = anndata.concat(loaded_anndatas, join='outer', merge='unique')
        # print(f"    Concatenated object has {concatenated_adata.n_obs} cells and {concatenated_adata.n_vars} features.")
        # concatenated_adata.write(os.path.join(data_directory, "FINAL_CONCATENATED_SCATAC.h5ad"))

if __name__ == "__main__":
    main()