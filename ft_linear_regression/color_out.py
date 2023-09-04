# color code
import sys

COL_RESET = '\x1b[0m'
COL_ORANGE = '\x1b[38:5:214m'
COL_GRNBLK = '\x1b[1;32;40m'
COL_GRNWHI = '\x1b[2;32;47m'
COL_BLUWHI = '\x1b[1;34;47m'
COL_BLUCYA = '\x1b[1;34;46m'
#Bolded
COL_BLURED = '\x1b[2;34;41m'
COL_REDWHI = '\x1b[2;31;47m'
COL_ERR = '\x1b[2;34;41m'

def print_title(mssg:str):
    print(f'{COL_BLUWHI}----------- ' + mssg 
          + f' -----------{COL_RESET}\n')
    
def print_check(mssg:str):
    print("✅",f'{COL_BLUWHI}' + mssg + f'{COL_RESET}')

def print_cross (mssg:str):
    print("❌", f'{COL_REDWHI}' + mssg + f'{COL_RESET}')

def print_result(mssg:str):
    # print(f'{COL_GRNWHI}' + mssg + f'{COL_RESET}')
    print(f'{COL_ORANGE}' + mssg + f'{COL_RESET}\n')

def print_stderr(mssg:str):
    print (f'{COL_ERR}' + mssg + f'{COL_RESET}', file=sys.stderr)

def print_status(mssg:str):
    print(f'{COL_GRNWHI}' + mssg + f'{COL_RESET}')

def print_comment(mssg:str):
    pass

def print_result(mssg:str):
    # print(f'{COL_GRNWHI}' + mssg + f'{COL_RESET}')
    print(f'{COL_ORANGE}' + mssg + f'{COL_RESET}')

def input_user(mssg:str) -> str:
    answer = input(f'{COL_GRNWHI}' + mssg + f'{COL_RESET}')
    return answer

def input_user_yes(mssg:str, pos_answers = ["y", "Y"], neg_answers = ["n", "N"] ) -> bool:
    try:
        answer = input(f'{COL_GRNWHI}' + mssg + f'(y / n) ? {COL_RESET}')
    except (EOFError):
        print_stderr("\nError : unexpected end of file !")
        return False
    return answer in pos_answers