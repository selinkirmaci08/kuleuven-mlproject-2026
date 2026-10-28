
# Multi-Agent Reinforcement Learning: Matrix Games and Knights Archers Zombies

Term project for the course [Machine Learning: Project](https://onderwijsaanbod.kuleuven.be/syllabi/e/H0T25AE.htm) at KU Leuven (2025-2026), by Selin Kırmacı and Marta Toca Carballo.

## What we built

**Learning dynamics in matrix games** ([`matrix-games/`](matrix-games))
- Implemented epsilon-greedy, Boltzmann and lenient Boltzmann Q-learning.
- Compared them on Stag Hunt, the Subsidy Game, Prisoner's Dilemma and Biased Rock-Paper-Scissors, and plotted learning trajectories against the replicator dynamics.
- Showed how leniency moves Stag Hunt and the Subsidy Game towards the cooperative, Pareto optimal outcome.

**Two cooperating archers in Knights Archers Zombies** ([`train_vector.py`](train_vector.py), [`submission_vector.py`](submission_vector.py))
- Trained two archers with PPO (Ray RLlib), one policy per archer (3-layer MLP each), to defend together against incoming zombies.
- Designed a 40-value observation per agent: each archer sees its own state and its partner's, plus the alignment, turning direction and distance of the four most urgent zombies.
- Added reward shaping for urgent zombies and randomly hid zombies during training, so the policy stays robust when the detector misses one.
- The policy reached a peak average reward of 48.7. Across all six distortion levels the agent scores 47 to 49, against 2 to 6 for a random agent.

**Zombie detection** ([`zombie_detection/`](zombie_detection))
- Generated a labelled dataset of 6,757 game frames over six visual distortion levels.
- Trained YOLO11n on clean frames first, then fine-tuned it on all distortion levels.
- Validation results: mAP50 0.975, mAP50-95 0.902, precision 0.997, recall 0.976.

---

## Course template

The rest of this README is the original setup guide from the course.

This repository contains the code to setup the final evaluation of the course "[Machine Learning: Project](https://onderwijsaanbod.kuleuven.be/syllabi/e/H0T25AE.htm)" (KU Leuven, Faculty of Engineering, Department of Computer Science, [DTAI Section](https://dtai.cs.kuleuven.be)).


## Summary

- Install all dependencies in `requirements.txt`
- Train two agents using an RL library. See `example_training_rllib.py` for an example of training simple agents using Ray RLLib.
- The KAZ environment is created by using the corresponding function in the `utils.py` file. 
- During your project, you can test your trained agent via the `evaluation.py` file. To know about the options, run `python3 evaluation.py -h`.
- For the final submission, you need to implement the interfaces in the file `submission.py` for the two archers (Task 3) environment. 
- Submit your own code on the departmental computers for the tournament (see below). See the file `submission_example_rllib.py` for an example.


## Use on departmental computers

The departmental computers will be used to run a tournament and submit your implementation (see detailed instructions below). You can also use these computers to train your agents. A tutorial to connect remotely via SSH can be found [here](ssh.md) and additional info is available on [the departmental web pages](https://system.cs.kuleuven.be/cs/system/wegwijs/computerklas/index-E.shtml).

You will see a personal directory in:

```
/cw/lvs/NoCsBack/vakken/H0T25A/ml-project/students/r1234567
```

*Note:* You do not have permission to view other students’ directories, so running `ls students` will not work. Instead, navigate directly to your own directory using `cd`.

There is an upper limit of 50MB on the disk space that you can use. Remote (ssh) users are also limited to 2GB of RAM.

PyGame, PettingZoo and other packages that you can use are pre-installed in a virtual environment, which can be activated using:

```
source /cw/lvs/NoCsBack/vakken/H0T25A/ml-project/venv/bin/activate
```

Since this virtual environment will be used to run the tournament, you should avoid language features that are not compatible with the installed Python version (3.12) or use packages that are not installed. All of PettingZoo's [butterfly](https://pettingzoo.farama.org/content/basic_usage/) dependencies are currently installed, as well as `torch==2.10.0` and `tensorflow==2.20.0`.


## Local installation

- It is recommended to use a newly-created virtual environment to avoid dependency conflicts.


- Install Pettingzoo with the additional requirements for the Butterfly environments

    ```
    pip install 'pettingzoo[butterfly]'
    ```

- Install SuperSuit, which will help managing your environments:

    ```
    pip install supersuit
    ```

- Your agents will be dependent on some RL library. Here we provide an example for installing Ray RLlib:

    ```
    pip install 'ray[rllib]'
    ```

- All dependencies are also listed in the `requirements.txt` file (`pip install -r requirements.txt`). Or in the `pyproject.toml` file if you want to use `uv`.

You can (visually) test your installation by running:

```
python3 evaluation.py -l random_agent.py -s --distortion=5
python3 evaluation.py -l submission.py --distortion=0 --episodes=1
```


## Zombie detection

An unseen test set of observations will be given to your agent. Your agent should return the bounding boxes of all the zombies it has found. Based on intersection over union, the average precision is computed. You can test the setup with an example data file in the `observation_data` folder and running:

```
python3 evaluation.py -l random_agent.py --zombies
```


## Tournament

The tournament will be played with agents that are available on the departmental computers. This will allow you to try your agents in the identical environment that is used by the tournament script. For this to work, you have to adhere to the following setup:

- Your agents implement the interface in the `submission.py` 
- The tournament code will scrape the entire directory provided for you on the departmental computers for the `submission.py` file. If multiple matching files are found, a random one will be used.
- Your agents should be ready to play in a few seconds, thus use pre-trained policies. An agent that is not responding after 10 seconds will forfeit the game.
- There is no timeout on the actions. The required speed is defined by the zombies that move down. Check your code on the departmental computers to get an idea of how fast your code runs. Or implement a timeout yourself to guarantee fast enough actions.

### Paths

Make sure you **do not use relative paths** in your implementation to load your trained model, as this will fail when running your agent from a different directory. Best practice is to retrieve the absolute path to the module directory:

```python
package_directory = os.path.dirname(os.path.abspath(__file__))
```

Afterwards, you can load your resources based on this `package_directory`:

```python
model_file = os.path.join(package_directory, 'models', 'mymodel.pckl')
```

## Submission using the Departmental Computers

To submit your agent, a copy of your code and agent needs to be available on the departmental computers in a directory assigned to you. Also the code to train your agent should be included.

The departmental computers have the `requirements.txt` packages already installed such that you can verify that your agent works. During the semester the tournament script will be run to play games between the (preliminary) agents that are already available. A tentative ranking will be shared.


## FAQ

### Where do I ask questions?

On the Toledo Discussion board or in the Q&A sessions.

### I found a mistake in the template or the examples in this repository

Submit a Github pull request.

### Installation cannot find Tensorflow

Tensorflow is only compatible with Python 3.9‒3.12.

On macOS you can use an older Python version by running these commands before the install script:

```
brew install python@3.12  # if using homebrew
virtualenv -p /usr/local/opt/python@3.12/bin/python3 venv
. ./venv/bin/activate
```

### Getting error during the installation of the requirements

Consider downgrading your python version to 3.12 or lower as some packages might not yet support the latest python version.

