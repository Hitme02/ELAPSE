import tarfile
import requests
import sys

print("Packaging files...")
try:
    with tarfile.open('paper.tar.gz', 'w:gz') as tar:
        tar.add('ELAPSE_paper.tex')
        tar.add('output/figures/fig1_entropy_n100.png')
        tar.add('output/figures/fig3_iee_n500.png')
        tar.add('output/figures/fig6_weights_n100.png')
        tar.add('output/figures/fig7_scaling.png')
        tar.add('output/figures/fig8_sensitivity.png')
except Exception as e:
    print(f"Failed to create tar.gz: {e}")
    sys.exit(1)

print("Uploading to latexonline.cc for compilation...")
try:
    with open('paper.tar.gz', 'rb') as f:
        files = {'file': ('paper.tar.gz', f, 'application/x-gzip')}
        params = {'target': 'ELAPSE_paper.tex', 'command': 'pdflatex'}
        response = requests.post("https://latexonline.cc/data", files=files, params=params, timeout=120)

    if response.status_code == 200:
        with open('ELAPSE_paper.pdf', 'wb') as f:
            f.write(response.content)
        print("Success! PDF saved as ELAPSE_paper.pdf")
    else:
        print(f"Compilation failed with status {response.status_code}.")
        print(response.text[:2000])
except Exception as e:
    print(f"Request failed: {e}")
