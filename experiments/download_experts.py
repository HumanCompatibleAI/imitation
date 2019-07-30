import urllib.request
import zipfile

experts_url = ("https://www.dl.dropboxusercontent.com/"
               "s/u3uuyd8nxs7obwp/expert_policies.zip")

experts_dest = "expert_models/"


def main(verbose=True):
  try:
    zip_path, _ = urllib.request.urlretrieve(experts_url)
  except urllib.error.URLError as e:
    if "SSL" in str(e):
      print(
        "Suggestion: If on macOS, run "
        r"`/Applications/Python\ 3.6/Install\ Certificates.command` "
        "(https://stackoverflow.com/a/13531310/1091722)")
    raise e

  if verbose:
    print("Downloaded expert_policies.zip")

  with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("expert_models")
    if verbose:
      print(f"Extracted the following experts to {experts_dest}:")
      zip_ref.printdir()


if __name__ == "__main__":
  main()
