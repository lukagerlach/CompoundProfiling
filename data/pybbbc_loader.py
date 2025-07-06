import sys
from pybbbc import BBBC021

def download_bbbc021(data_root: str = "/scratch/cv-course2025/group8") -> None:
    """Downloads the complete BBBC021 dataset.

    Args:
        data_root: Root directory where the dataset will be stored.
        
    Note:
        We are using the pybbbc library to download the dataset. This
        function is simply a wrapper around the `BBBC021.download`
        method.
    
    Note:
        If you  are using RAMSES, you can use the default path,
        the data should be there already.
        
    Dataset Structure:
        After download, the directory structure should look like::
        
            data_root/raw/
            ├── images/
            │   ├── Week1_22123/
            │   ├── Week1_22141/
            │   ├── Week1_22161/
            │   └── ... (all experimental weeks)
            └── some_metadata_file.csv
            
    Example:
        >>> # Download to default location
        >>> download_bbbc021()
        
        >>> # Download to custom location
        >>> download_bbbc021("/path/to/my/data/bbbc021")
        
    References:        
        Dataset: https://bbbc.broadinstitute.org/BBBC021
        pybbbc docs: https://github.com/giacomodeodato/pybbbc
    """
    
    BBBC021.download(root_path=data_root)  # Downloads the dataset files
    print(f"BBBC021 dataset downloaded and extracted to {data_root}.")
    
def preprocess_bbbc021(data_root: str = "/scratch/cv-course2025/group8") -> None:
    """
    Preprocess the BBBC021 dataset. More information on the preprocessing
    can be found in the pybbbc documentation.
    
    Args:
        data_root: Root directory where the raw dataset is stored.
    """
    # Create the dataset structure
    BBBC021.make_dataset(root_path=data_root)
    print(f"BBBC021 dataset preprocessed and ready for use at {data_root}.")

if __name__ == "__main__":
    #download_bbbc021()
    #preprocess_bbbc021()
    #print("BBBC021 dataset is ready for use.")

    # to be able to specify a path
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "/scratch/cv-course2025/group8"

    download_bbbc021(data_root)
    preprocess_bbbc021(data_root)
    print(f"BBBC021 dataset is ready for use at {data_root}/processed")

# python3 data/pybbbc_loader.py /custom/data/path