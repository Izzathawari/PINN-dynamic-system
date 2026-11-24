from train_model import train_model

def main ():

    total_loss = 0
    for i in range(3):
        model , history, test_loss = train_model()

        total_loss = test_loss + total_loss

    print(f"AVERAGE LOSS * {total_loss}")

if __name__ == "__main__":

    main()          