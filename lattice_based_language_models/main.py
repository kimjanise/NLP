### Using SRILM to generate a Language Model

!apt-get update
!apt-get install build-essential gawk
!wget web2.qatar.cmu.edu/~ko/11411/srilm-1.7.2.tar.gz
!tar -xvzf srilm-1.7.2.tar.gz
%cd /content/
%env SRILM=/content/
!./sbin/machine-type
!make World MACHINE_TYPE=i686-m64
%cd bin/i686-m64/

!./ngram -help
!pip install gdown
!gdown --id 1RnpaF0vV7m1AFPXaNG-2BPj0FQuQureR -O data.txt
!head -100 /content/bin/i686-m64/data.txt

l = 5000
i = 0
data = open('/content/bin/i686-m64/data.txt','r')
train = open('train.txt',"w")
test = open('test.txt',"w")

for d1 in data:
    if (i < l):
      test.write(d1)
    else:
      break
    i = i + 1

for d2 in data:
    train.write(d2)

train.close()
test.close()

!./ngram-count -text '/content/bin/i686-m64/train.txt' -lm '/content/bin/i686-m64/corpus.srilm'

### Using Lattice Tool to Score Lattices

def lattice_generation(sentence):

    sentence = sentence.strip()
    name = "example"  #name name
    #w_1 to w_N:
    w = str() #initializes w
    N, counter = 1, 0
    w+="<s> "

    max_char = 0
    for part in sentence:
      if (part.isdigit()|part.isalpha()):
        max_char = max_char + 1

    temp_count = 0
    fin = ""
    for part in sentence:
        fin = part
        if (part.isdigit()|part.isalpha() & (temp_count < (max_char - 1))): #if either a number of a letter
            w += (part+" # ")
            N += 2
            temp_count = temp_count + 1
    if (fin.isdigit()|fin.isalpha()):
      w = w + fin
      N = N + 1
    w = w + " </s>"
    N = N + 1
    splitw = w.split(" ")

    f = N - 1

    T = 0
    for ch in splitw:
      if (ch == "<s>"):
        T += 1
      elif (ch == "</s>"):
        T += 0
      elif (ch == "#"):
        T += 1
      else:
        T += 2

    paths, finalString = list(), str()

    count = 0
    for ch in splitw:
      if (ch == "<s>"):
        paths.append("0 1 0")
      elif (ch == "</s>"):
        paths.append(f'{count} {count + 1} 0')
      elif (ch == "#"):
        paths.append(f'{count} {count + 1} 0')
      else:
        paths.append(f'{count} {count + 1} 0')
        paths.append(f'{count} {count + 2} 0')
      count = count + 1

    for p in paths:
      s = p + '\n'
      finalString = finalString + (s)

    lattice = (f"name {name}\nnodes {N} {w}\ninitial {0}\nfinal {f}\ntransitions {T}\n{finalString.strip()}")
    return lattice #lattice file for sentence

!mkdir out

!pwd

!./lattice-tool -in-lattice 'path to your lattice file' -lm 'path to your lm' -nbest-decode 1 -order 5  -out-nbest-dir 'path to your output file'

import gzip

def extract_from_output_file(output_dir):
    with gzip.open(output_dir, 'rt') as gz_file:
        content = gz_file.read()

    start_idx = content.find('<s>') + 4  # +4 to move after the <s> tag
    end_idx = content.find('</s>')

    extracted_content = content[start_idx:end_idx].strip()

    return extracted_content.replace(' ', '')

predict_out = extract_from_output_file('path-to-out-directory/example.gz')



