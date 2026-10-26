import unittest
import numpy as np
from app.metric_learning import identificar, prototipos


class RegrasTeste(unittest.TestCase):
    def test_galeria_vazia(self):
        self.assertEqual(identificar(np.array([1., 0.]), {}, .4)[0], 'Desconhecido')

    def test_prototipo_normalizado(self):
        protos = prototipos({'Ana': np.array([[1., 0.], [1., 0.]])})
        np.testing.assert_allclose(protos['Ana'], [1., 0.])

    def test_threshold(self):
        protos = {'Ana': np.array([1., 0.])}
        self.assertEqual(identificar(np.array([0., 1.]), protos, .4)[0], 'Desconhecido')
        self.assertEqual(identificar(np.array([1., 0.]), protos, .4)[0], 'Ana')


if __name__ == '__main__':
    unittest.main()
