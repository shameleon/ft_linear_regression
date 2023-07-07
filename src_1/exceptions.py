

class InvalidMileageRangeException(Exception):
    """ Exception raised for errors in the input Mileage"""

    def __init__(self, mileage, message="Invalid mileage : out of range"):
        self.mileage = mileage
        self.message = message
        super().__init__(self.message)


class InvalidPriceException(Exception):
    """ Price must be >= 0 """
    pass


def main():
    intercept = 9000.0
    slope = -0.25
    try:
        mileage = int(input("Please enter a mileage : "))
        if not 0 <= mileage < 1E6:
            raise InvalidMileageRangeException(mileage)
    except (ValueError):
        print("Error : Value Error, not an integer number")
    else :
        price = slope * mileage + intercept
        try:
            assert price >= 0
        except:
            price = 0
            raise InvalidPriceException()
        finally:
            print("Predicted price :", price)

if __name__ == "__main__":
    """testing exceptions"""
    main()