import numpy as np
import logging

logger = logging.getLogger(__name__)
TRAPEZOID_INTEGRAL = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class RulHandler:
    def prepare_y_future(
        self,
        battery_names,
        battery_n_cycle,
        y_soh,
        current,
        time,
        capacity_threshold=None,
        allow_negative_future=False,
        capacity=None,
    ):
        cycle_lenght = current.shape[1]
        battery_range_step = [x * cycle_lenght for x in battery_n_cycle]
        logger.info("battery step: {}".format(battery_n_cycle))
        logger.info("battery ranges: {}".format(battery_range_step))

        if capacity is None:
            battery_nominal_capacity = [
                float(name.split("-")[2]) for name in battery_names
            ]
        else:
            battery_nominal_capacity = [capacity for name in battery_names]

        current = current.ravel()
        time = time.ravel()
        capacity_integral_train = []
        a = 0  # Integration over the every battery
        for battery_index, b in enumerate(battery_range_step):
            logger.info("processing range {} - {}".format(a, b))
            integral_sum = 0
            pre_i = a
            for i in range(a, b, cycle_lenght):
                integral = TRAPEZOID_INTEGRAL(
                    y=current[pre_i:i][current[pre_i:i] > 0],
                    x=time[pre_i:i][current[pre_i:i] > 0],
                )
                integral_sum += integral
                pre_i = i
                capacity_integral_train.append(
                    integral_sum / battery_nominal_capacity[battery_index]
                )
            a = b
        capacity_integral_train = np.array(capacity_integral_train)
        logger.info("Train integral: {}".format(capacity_integral_train.shape))
        # capacity_integral_train -->(Total cycles per N batteties, 1)
        y_future = []
        a = 0
        for battery_index, b in enumerate(battery_n_cycle):
            logger.info("processing range {} - {}".format(a, b))
            if capacity_threshold is None:
                index = b - 1
            else:
                index = (
                    np.argmax(
                        y_soh[a:b]
                        < capacity_threshold[battery_nominal_capacity[battery_index]]
                    )
                    + a
                )
                if index == a:
                    index = b - 1
            logger.info("threshold index: {}".format(index))
            for i in range(a, b):
                if not allow_negative_future:
                    y = (
                        capacity_integral_train[index] - capacity_integral_train[i]
                        if i < index
                        else 0
                    )
                else:
                    y = capacity_integral_train[index] - capacity_integral_train[i]
                y_future.append(y)
            a = b
        y_future = np.array(y_future)
        logger.info("y future: {}".format(y_future.shape))

        y_with_future = np.column_stack((capacity_integral_train, y_future))
        logger.info("y with future: {}".format(y_with_future.shape))
        return y_with_future
