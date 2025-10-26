def running_mean_cal(mean_previous,inp,iterator):
    """AI is creating summary for running_mean_cal

    Args:
        mean_previous ([float]): previously calculated mean value.
        inp ([float]): new input given.
        iterator ([float]): Current iterator

    Returns:
        [float]: New mean of the data.
    """

    current_mean = round(mean_previous + ((inp-mean_previous)/iterator),4)
    
    return current_mean

if __name__ =="__main__":
    """
     Running mean is function is used to calculate the mean of streaming data, 
     without need to store the data or go over the entire records at once. Also 
     used when need to calculates mean by sampling of data.

    """

    # Define the constant 
    iterator:int = 1
    mean_previous:float = 0

    # Start the loop
    while True:

        # Take the input 
        inp = input("Enter the integer value or 'q' to quit the program, \n > ")

        # Check if quit is given
        if str(inp) =="q":
            break
        
        # Check the given input is numeric
        elif inp.isnumeric():
            
            # Calculate the running mean by required paramters
            result = running_mean_cal(mean_previous,float(inp),iterator)

            # Print the results
            print(f"Running mean is {result}")

            # Update the iterators and assignee the mean as previous mean.
            iterator += 1
            mean_previous = result

        # Ask user to give integer value.
        else:
            print("Please enter the integer value.")
