"""Run ad-hoc experiments on the command line.

Run ``python3 CPPL_test_run.py -to 300 -d problem_instance_directory -p pws -s solver_name``
"""
import argparse
from time import time
from CPPL_class.cppl_algo import CPPLAlgo
from preselection import regret_minimizing_algorithm


def _main():
    parser = argparse.ArgumentParser(description="Start CPPL Tournament")
    preselection_algorithm = {
        algorithm.__name__: algorithm for algorithm in regret_minimizing_algorithm
    }
    preselection_algorithm_choices = " ".join(preselection_algorithm.keys())

    ################## CPPL Arguments ##################
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="No Problem Instance Directory given",
        help="Directory for instances",
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="No solver chosen",
        help="Solver/math. model as .py",
    )
    parser.add_argument(
        "-to",
        "--timeout",
        type=int,
        default=300,
        help="""Stop solving single instance after (int) seconds [300]""",
    )
    parser.add_argument(
        "-nc",
        "--contenders",
        type=int,
        default=30,
        help="The number of contenders [30]",
    )
    parser.add_argument(
        "-keeptop",
        "--top_to_keep",
        type=int,
        default=2,
        help="""Number of top contenders to get chosen for game [2]""",
    )
    parser.add_argument(
        "-p", "--pws", type=str, default=None, help="Custom Parameter Genome []"
    )
    parser.add_argument(
        "-usedata",
        "--data",
        type=str,
        default=None,
        help="""Type y if prior gene and score data should be used []""",
    )
    parser.add_argument(
        "-experimental",
        "--exp",
        type=str,
        default=None,
        help="""Type y if prior gene and score data should be experimental (Pool_exp.txt) []""",
    )
    parser.add_argument(
        "-ch",
        "--chance",
        type=int,
        default=25,
        help="""Chance to replace gene randomly in percent (int: 0 - 100) [25]""",
    )
    parser.add_argument(
        "-m",
        "--mutate",
        type=int,
        default=10,
        help="""Chance for mutation in crossover process in percent (int: 0 - 100) [10]""",
    )
    parser.add_argument(
        "-k",
        "--kill",
        type=float,
        default=5,
        help="""Contenders with a variance higher than this are killed and replaced (float) [5]""",
    )
    parser.add_argument(
        "-tn",
        "--train_number",
        type=float,
        default=None,
        help="""How many of the first instances are to be trained on before starting (int) [None] """,
    )
    parser.add_argument(
        "-tr",
        "--train_rounds",
        type=float,
        default=0,
        help="""How many rounds are the first -tn instances to be trained on (int) [1] """,
    )
    parser.add_argument(
        "-fo",
        "--file_order",
        type=str,
        default="ascending",
        help="""Specify the order by which the problem instances are solved""",
    )
    parser.add_argument(
        "-nc_pca_f",
        "--nc_pca_f",
        type=int,
        default=7,
        help="""Number of the dimensions for the PCA of the instance features """,
    )
    parser.add_argument(
        "-nc_pca_p",
        "--nc_pca_p",
        type=int,
        default=10,
        help="""Number of the dimensions for the PCA of the parameter (features) """,
    )
    parser.add_argument(
        "-jfm",
        "--jfm",
        type=str,
        default="polynomial",
        help="""Mode of the joined feature map""",
    )
    parser.add_argument(
        "-omega",
        "--omega",
        type=float,
        default=0.001,
        help="""Omega parameter for CPPL""",
    )
    parser.add_argument(
        "-gamma", "--gamma", type=float, default=1, help="""Gamma parameter for CPPL"""
    )
    parser.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        default=0.2,
        help="""Alpha parameter for CPPL""",
    )
    parser.add_argument(
        "-tfn",
        "--times_file_name",
        type=str,
        default="Times_per_instance_CPPL",
        help="""Name of the file which the times needed to solve instances are tracked in""",
    )
    parser.add_argument(
        "-pl",
        "--paramlimit",
        type=float,
        default=100000,
        help="""Limit for the possible absolute value of a parameter for it to be normed to log space before CPPL computation""",
    )
    parser.add_argument(
        "-bp",
        "--baselineperf",
        type=bool,
        default=False,
        help="""Set to true if only default parameterization should run""",
    )

    ################## Preselection Arguments ##################
    parser.add_argument(
        "-a",
        "--algorithms",
        metavar="ClassName",
        default=preselection_algorithm.keys(),
        help=f"Algorithm for selecting the optimal interim subset. (default: {preselection_algorithm_choices})",
        choices=preselection_algorithm.keys(),
    )

    args, unknown = parser.parse_known_args()
    cppl_run = CPPLAlgo(
        args=args
    )  # TODO add an experiment script for running CPPL with different reselection bandits
    cppl_run.run()


if __name__ == "__main__":
    start = time()
    _main()
    end = time()
    print("Execution time: ", end-start)
