# color code
import sys

""" https://talyian.github.io/ansicolors/ """
COL_RESET = '\x1b[0m'
COL_ORANGE = '\x1b[38:5:208m'
COL_TURQU = '\x1b[38;5;45m'
COL_FTLIN = '\x1b[48:5:208m'
COL_GRNBLK = '\x1b[1;32;40m'
COL_GRNWHI = '\x1b[2;32;47m'
COL_BLUWHI = '\x1b[2;34;47m'
COL_BLUCYA = '\x1b[1;34;46m'
#Bolded
COL_BLUWHI = '\x1b[1;34;47m'
COL_BLURED = '\x1b[2;34;41m'
COL_REDWHI = '\x1b[1;31;47m'
COL_ERR = '	\x1b[38;5;9m'
COL_QUERY = '\x1b[2;37;40m'
COL_QUERY2 = '\x1b[2;37;47m'
COL_ASKKM = '\x1b[2;34;43m'

def print_title(mssg:str):
    print(f'\n              {COL_FTLIN}ft_linear regression{COL_RESET}')
    print(f'{COL_BLUWHI}-----------  ' + mssg 
          + f'  -----------{COL_RESET}\n')

def print_title2(mssg:str):
    print(f'\n{COL_BLUCYA}----------- ' + mssg 
          + f' -----------{COL_RESET}\n')
    
def print_title3(mssg:str):
    print(f'{COL_BLUWHI}----------- ' + mssg 
          + f' -----------{COL_RESET}\n')

def print_check(mssg:str):
    print("✅",f'{COL_BLUWHI}' + mssg + f'{COL_RESET}\n')

def print_cross (mssg:str):
    print("❌", f'{COL_REDWHI}' + mssg + f'{COL_RESET}\n')

def print_result(mssg:str):
    # print(f'{COL_GRNWHI}' + mssg + f'{COL_RESET}')
    print(f'{COL_ORANGE}' + mssg + f'{COL_RESET}\n')

def print_stderr(mssg:str):
    print (f'{COL_ERR}' + mssg + f'{COL_RESET}', file=sys.stderr)

def print_status(mssg:str):
    print(f'{COL_FTLIN}' + mssg + f'{COL_RESET}')

def print_comment(mssg:str):
    print(f'{COL_GRNWHI}' + mssg + f'{COL_RESET}')

def input_user_str(mssg:str) -> str:
    answer = input(f'{COL_QUERY}' + mssg + f'{COL_ORANGE}   ')
    print(f'{COL_RESET}')
    return answer

def input_user_yes(mssg:str, pos_answers = ["y", "yes"] ) -> bool:
    try:
        answer = input(f'{COL_QUERY}' + mssg + f' (y / n) ? {COL_RESET}')
    except (EOFError):
        print_stderr("\nError : unexpected end of file !")
        return False
    return answer.lower() in pos_answers