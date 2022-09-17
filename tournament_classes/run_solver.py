from subprocess import Popen, PIPE
import subprocess
from typing import List, Tuple, Union, Any, Iterable


def start(params: List, time_limit: int, filename: str, solver: str) -> Popen:
    """
    Sub-routine of sub-process to solve problem instances using different solvers.

    Parameters
    ----------
    params : List
        Set of different combination of parameters to be used to solve the problem instances by the solver.
    time_limit : int
        Maximum time limit a solver can run, after this threshold the execution will stop.
    filename : str
        The problem instance file name.
    solver : str
        The name if the solver.

    Returns
    -------
    proc : Popen
        The subprocess.Popen object to run the solver with the given parameters in the sub-routine of parallel threads.
    """
    if solver == "cadical":

        proc = subprocess.Popen(
            [
                "solvers/./cadical",
                "--no-witness",
                *params,
                "-t",
                str(time_limit),
                f"{filename}",
            ],
            stdout=PIPE,
        )
        return proc

    elif solver == "glucose":

        proc = subprocess.Popen(
            [
                "solvers/./glucose_static",
                "-cpu-lim=" + str(time_limit),
                *params,
                f"{filename}",
            ],
            stdout=PIPE,
        )
        return proc

    elif solver == "cplex":

        dir_file = list(filename.split("/"))
        directory = dir_file[0]
        filename = dir_file[1]

        proc = subprocess.Popen(
            [
                "python3",
                "solvers/cplex_solve.py",
                "-d",
                directory,
                "-f",
                f"{filename}",
                "-to",
                str(time_limit),
                "-g",
                str(params),
            ],
            stdout=PIPE,
        )

        return proc


def check_output(line: str, interim: List, solver: str) -> Union[List, str]:
    """
    Check the output of the solver on the problem instance.

    Parameters
    ----------
    line : str
        Each line in the problem instance file.
    interim : List
        Interim list of the threads.
    solver : str
        Solver's name used to solve the problem instances.

    Returns
    -------
    Union[List, str]
        The interim output of the subprocess.
    """
    if solver == "cadical":

        if line != b"":
            b = str(line.strip())
            # Check for progress
            if b[2] == "c":
                if len(b) > 4:
                    b = list(b.split(" "))
                    b = list(filter(lambda a: a != "", b))
                    if len(b) > 13:
                        interim = [
                            int(b[4]),
                            int(b[5]),
                            int(b[6]),
                            int(b[7]),
                            int(b[11]),
                            int(b[13][:-2]),
                        ]

                        return interim
        else:
            return "No output"

    elif solver == "glucose":

        if line != b"":
            b = str(line.strip())
            if b[2] == "c":
                b = list(b.split("  "))
                b = list(filter(lambda a: a != "", b))
                if len(b) > 11 and len(b[11]) > 5:
                    interim = [
                        int(b[2][1:]),
                        int(b[3][:-2]),
                        int(b[4][1:]),
                        int(b[5][1:]),
                        int(b[6][1:-2]),
                        int(b[7]),
                        int(b[8]),
                        int(b[9]),
                        int(b[10][:-1]),
                        float(b[11][:-4]),
                    ]
                    return interim
        else:
            return "No output"

    elif solver == "cplex":

        if line != b"":
            currentobj = interim[0]
            gap = interim[1]
            bestbound = interim[2]
            numbernodes = interim[3]
            b = str(line.strip())
            if b[2:4] == "CI":
                b = b[6:-1]
                currentobj = float(b)
            elif b[2] == "G":
                a = float(b[6:-1])
                gap = int(a)
            elif b[2] == "B":
                b = b[2:-1]
                bestbound = float(b[4:])
            elif b[2:4] == "CN":
                a = float(b[7:-1])
                numbernodes = a
            interim = [currentobj, gap, bestbound, numbernodes]

            return interim
        else:
            return "No output"


def check_if_solved(
    line: str,
    results: List[int],
    proc: Popen,
    event: List[int],
    non_nlock_read: Any,
    solver: str,
) -> Tuple[Iterable, Iterable, Any]:
    """
    Check if the solver has solved the problem instance. If yes, return the event and the result.

    Parameters
    ----------
    line : str
        Each line in the problem instance file.
    results : List[int]
        Current result of the process.
    proc : Union[List, str]
        The subprocess object.
    event : List[int]
        The event in the running thread.
    non_nlock_read : Any
        A method to check the output of the process.
    solver : str
        Solver's name used to solve the problem instances.

    Returns
    -------
    If the instance problem is solved then,
        results : List
            The results of the problem instance.
        event : List
            The event list including the one which finished first.
    else,
         str
            Indicating 'No output'
    """
    if solver == "cadical":

        if line != b"":
            b = str(line.strip())
            if b[2] == "s":
                results[0] = 1
                time_not_given = True
                while time_not_given:
                    line = non_nlock_read(proc.stdout)
                    b = str(line.strip())
                    if len(b) > 4:
                        if b[4:17] == "total process":
                            b = list(b.split("  "))
                            b = list(filter(lambda a: a != "", b))
                            results[1] = float(b[1])
                            time_not_given = False

                event[0] = 1
                proc.stdout.close()

            return results, event
        else:
            return "No output"

    elif solver == "glucose":

        if line != b"":
            b = str(line.strip())
            if len(b) > 4:
                if b[4] == "C":
                    b = list(b.split("  "))
                    b = b[7].split()
                    results[1] = float(b[1])
                    result_not_given = True
                    while result_not_given:
                        line = non_nlock_read(proc.stdout)
                        b = str(line.strip())
                        if len(b) > 4:
                            if b[2] == "s" and b[4] != "I":
                                results[0] = 1
                                result_not_given = 0
                                event[0] = 1
                                proc.stdout.close()

            return results, event
        else:
            return "No output"

    elif solver == "cplex":

        if line != b"":
            b = str(line.strip())
            if b[2] == "[":
                b = b[2:-3]
                b = list(b.split(", "))
                results[:] = [float(b[0][1:]), float(b[1][:-1])]
                event[0] = 1
                proc.stdout.close()

            return results, event
        else:
            return "No output"
