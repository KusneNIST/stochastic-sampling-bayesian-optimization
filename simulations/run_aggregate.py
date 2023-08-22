
if __name__ == "__main__":

    import aggregate
    import argparse
    import os

    
    parser = argparse.ArgumentParser()
    parser.add_argument('path')

    args = parser.parse_args()

    for objective in os.listdir(args.path):
        if os.path.isdir(os.path.join(args.path, objective)):
            for config in os.listdir(os.path.join(args.path, objective)):
                if os.path.isdir(os.path.join(args.path, objective, config)):

                    os.system(
                        'python aggregate.py --step 10 --log-space {} regret.csv 0'.format(
                            os.path.join(args.path, objective, config)
                        )
                    )

                    os.system(
                        'python aggregate.py --step 5 {} hyperparams.csv 2'.format(
                            os.path.join(args.path, objective, config)
                        )
                    )
