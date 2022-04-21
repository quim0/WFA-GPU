import argparse

def check_cigar_sequences(score, cigar_ops, cigar_reps, pattern, text):
    text_pos = 0
    pattern_pos = 0

    for idx, op in enumerate(cigar_ops):
        reps = cigar_reps[idx]
        for _ in range(reps):
            if op == 'M':
                if pattern[pattern_pos] != text[text_pos]:
                    return False
                pattern_pos += 1
                text_pos += 1
            elif op == 'X':
                if pattern[pattern_pos] == text[text_pos]:
                    return False
                pattern_pos += 1
                text_pos += 1
            elif op == 'I':
                text_pos += 1
            elif op == 'D':
                pattern_pos += 1
            else:
                print(f"Invalid op {op}")
                return False

    if (pattern_pos != len(pattern)) or (text_pos != len(text)):
        return False

    return True
    

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*', help='Input files')
parser.add_argument('-g', '--penalties', required=True, help='x,o,e')

args = parser.parse_args()
if not args.files:
    print('No input files... Aborting.')
    quit()

X,O,E = map(int, args.penalties.split(','))

for f in args.files:
    results_file = f
    try:
        with open(results_file, 'r') as f:
            correct = 0
            incorrect = 0
            for line_num, line in enumerate(f):
                line = line.rstrip()
                elements = line.split()
                if len(elements) < 2 or len(elements) > 4:
                    raise ValueError('Invalid results file.')

                score = abs(int(elements[0]))
                cigar = elements[1]

                cigar_tmp = cigar.replace('M', ' ')
                cigar_tmp = cigar_tmp.replace('X', ' ')
                cigar_tmp = cigar_tmp.replace('I', ' ')
                cigar_tmp = cigar_tmp.replace('D', ' ')
                try:
                    cigar_reps = list(map(int, cigar_tmp.split()))
                except ValueError:
                    print(f"Invalid op at CIGAR {line_num}")
                    incorrect += 1
                    continue

                ops = []
                for e in cigar:
                    if e in ['M', 'X', 'I', 'D']:
                        ops.append(e)

                is_correct = True
                if len(elements) == 4:
                    pattern = elements[2]
                    text = elements[3]
                    ok = check_cigar_sequences(score, ops, cigar_reps, pattern, text)
                    if not ok:
                        is_correct = False
                        #print(f"CIGAR {line_num} do not fit the pattern and text.")

                # Calculate score
                cigar_score = 0
                for idx, op in enumerate(ops):
                    reps = cigar_reps[idx]
                    if op == 'M':
                        continue
                    elif op == 'X':
                        cigar_score += X * reps
                    elif op in ['I', 'D']:
                        cigar_score += O + E * reps

                if cigar_score != score:
                    is_correct = False

                if not is_correct:
                    #print(f'Incorrect CIGAR or score at {line_num}!')
                    #print(f'\tCIGAR:       {cigar}')
                    #print(f'\tScore:       {score}')
                    #print(f'\tCIGAR score: {cigar_score}')
                    incorrect += 1
                else:
                    correct += 1

                if (line_num % 1000000) == 0 and line_num > 0:
                    pass
                    #print(f'({results_file}) {line_num}: correct={correct}, incorrect={incorrect}')
            print(f"({results_file}) Correct={correct}, Incorrect={incorrect}")
    except (FileNotFoundError, IsADirectoryError) as e:
        print('Error opening file')
