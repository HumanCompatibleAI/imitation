import urllib
import zipfile

EXPERTS_URL = ("http://www.dl.dropboxusercontent.com/"
               "s/hww9n76d242433c/stable_baselines_expert_models.zip")

def main():
  zip_path, _ = urllib.urlretrieve(EXPERTS_URL, dest_zip_path)
  with zipfile.ZipFile(zip_path,"r") as zip_ref:
    zip_ref.extractall("expert_models")


if __name__ == "__main__":
  main()
