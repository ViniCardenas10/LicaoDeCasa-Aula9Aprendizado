#include "procimagem.h"

Mat_<float> deslocaImagem(Mat_<float> img, int x0, int y0)
{
    Mat_<float> res;
    Mat trans_mat = (Mat_<float>(2, 3) << 1, 0, x0, 0, 1, y0);
    warpAffine(img, res, trans_mat, img.size());
    return res;
}

int main()
{
    MNIST mnist(28, true, false);
    mnist.le("/home/vinicius/Documentos/Vinicius/CodigosAula/Inteligentes/MNIST/MNIST_ORG");
    double t1 = tempo();
    int n_aug = 5;
    Mat_<float> ax_aug(mnist.na * n_aug, mnist.ax.cols);
    Mat_<float> ay_aug(mnist.na * n_aug, 1);
    int k = 0;
    for (int i = 0; i < mnist.na; i++)
    {
        Mat_<float> img = mnist.ax.row(i).reshape(1, mnist.nlado);
        float label = mnist.ay(i, 0);
        img.reshape(1, 1).copyTo(ax_aug.row(k));
        ay_aug(k++, 0) = label;
        deslocaImagem(img, 0, -1).reshape(1, 1).copyTo(ax_aug.row(k));
        ay_aug(k++, 0) = label;
        deslocaImagem(img, 0, 1).reshape(1, 1).copyTo(ax_aug.row(k));
        ay_aug(k++, 0) = label;
        deslocaImagem(img, -1, 0).reshape(1, 1).copyTo(ax_aug.row(k));
        ay_aug(k++, 0) = label;
        deslocaImagem(img, 1, 0).reshape(1, 1).copyTo(ax_aug.row(k));
        ay_aug(k++, 0) = label;
    }
    flann::Index ind(ax_aug, flann::KDTreeIndexParams(64));
    double t2 = tempo();
    Mat_<int> matches(mnist.nq, 1);
    Mat_<float> dists(mnist.nq, 1);
    ind.knnSearch(mnist.qx, matches, dists, 1, flann::SearchParams(256));

    for (int l = 0; l < mnist.qx.rows; l++)
    {
        mnist.qp(l) = ay_aug(matches(l, 0), 0);
    }
    double t3 = tempo();
    float taxa_erro = 100.0 * mnist.contaErros() / mnist.nq;
    printf("Erros = %10.2f%%\n", taxa_erro);
    printf("Tempo de treinamento: %f segundos\n", t2 - t1);
    printf("Tempo de predicao: %f segundos\n", t3 - t2);
    int n_erros = mnist.contaErros();
    if (n_erros > 0)
    {
        int linhas_img = min(10, (n_erros / 10) + 1);
        int cols_img = min(10, n_erros);

        Mat_<uchar> img_erros = mnist.geraSaidaErros(linhas_img, cols_img);
        imwrite("erros_flann.png", img_erros);
        printf("A imagem 'erros_flann.png' foi salva com os recortes dos erros de classificacao.\n");
    }

    return 0;
}