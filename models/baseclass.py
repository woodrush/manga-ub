from abc import ABCMeta, abstractmethod
import numpy as np

class BenchmarkModel(metaclass=ABCMeta):
    image_by_impath = False

    @abstractmethod
    def run_q_a_raw(self, image, query):
        pass

    @staticmethod
    @abstractmethod
    def post_process_output(s):
        pass

    @abstractmethod
    def run_q_a_raw_multiple(self, impathlist, query):
        pass

    @staticmethod
    def image_is_null(image):
        return image is None or image == "n/a" or (type(image) == float and np.isnan(image)) or not image

    def run_q_a(self, image, query):
        raw_response = self.run_q_a_raw(image=image, query=query)
        return raw_response, self.post_process_output(raw_response)

    def run_q_a_multiple(self, impathlist, query):
        raw_response = self.run_q_a_raw_multiple(impathlist=impathlist, query=query)
        return raw_response, self.post_process_output(raw_response)
