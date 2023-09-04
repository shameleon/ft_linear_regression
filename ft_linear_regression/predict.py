import numpy as np
from my_colors import *
from PredictPriceClass import PredictPriceFromModel

def intro():
    print(f'\n{COL_BLUWHI}----------- PREDICT A CAR PRICE -----------{COL_RESET}\n\n')

def main():
    intro()
    model_prediction = PredictPriceFromModel("./gradient_descent_model/theta.csv")
    # test_predict_class(model_prediction)
    continue_loop = True
    while(continue_loop):
        model_prediction.ask_for_mileage()
        # print(model_prediction)
        try:
            in_str = input('\x1b[2;37;40m\npress [enter] to continue, [no] or [q] to quit\x1b[0m')
        except (EOFError):
            print("\nError : unexpected end of file")
        else:
            if (in_str in ["N", "No", "n", "no", "q", "Q", "exit", "quit", "QUIT"]):
                continue_loop = False


if __name__ == "__main__":
    main()

    """
    add line result to file """