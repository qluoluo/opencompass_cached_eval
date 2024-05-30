import os
fp1 = 'opencompass/models/myModel/flash_utils_v2/AttnCache.py'
fp2 = 'opencompass/models/myModel/flash_utils_v2/AttnCache_new.py'

def compare_file(fp1, fp2):
    with open(fp1, 'r') as f1, open(fp2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        # if len(lines1) != len(lines2):
        #     return False
        for i in range(len(lines1)):
            if lines1[i] != lines2[i]:
                print(i)
    return True

if __name__ == '__main__':
    print(compare_file(fp1, fp2))