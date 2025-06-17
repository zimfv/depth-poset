import time


class Timer:
    """
    Context manager for timing code execution, with support for intermediate checkpoints.

    Example usage::

        with Timer() as t:
            # some code
            t.checkpoint()
            # more code
        print(f"Duration: {t.get_duration()} seconds")

    Attributes
    ----------
    checkpoints : list of float
        List of timestamps recorded at each checkpoint.
    duration : float
        Total duration between the first and last checkpoint, available after exiting the context.
    """

    def __enter__(self):
        """
        Start the timer and initialize the first checkpoint.

        Returns
        -------
        Timer
            The instance itself, with the first checkpoint recorded.
        """
        self.checkpoints = [time.perf_counter()]
        return self

    def checkpoint(self):
        """
        Record the current time as a new checkpoint.
        """
        self.checkpoints.append(time.perf_counter())

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the timer when exiting the context, recording the final checkpoint
        and computing the total duration.

        Parameters
        ----------
        exc_type : type
            Exception type, if an exception was raised.
        exc_val : Exception
            Exception instance, if an exception was raised.
        exc_tb : traceback
            Traceback object, if an exception was raised.
        """
        self.checkpoint()
        self.duration = self.checkpoints[-1] - self.checkpoints[0]

    def elapsed(self):
        """
        Return the time elapsed since the last checkpoint.

        Returns
        -------
        float
            Time in seconds since the most recent checkpoint.
        """
        return time.perf_counter() - self.checkpoints[-1]

    def get_duration(self):
        """
        Get the total duration measured between the first and last checkpoint.
        If the context is still open, returns the time since the last checkpoint.

        Returns
        -------
        float
            Duration in seconds.
        """
        try:
            return self.duration
        except NameError:
            return self.elapsed()
