#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/text.hpp>
#include <iostream>


cv::CascadeClassifier patenteIdentifier;

int main(int argc, const char** argv)
{

    cv::CommandLineParser parser(argc, argv,
        "{image_path|./test.jpg|}"
        "{camera|0|Camera device number.}");
    parser.about("\nDemostracion de clasificador entrenado para detectar revolveres\n\n");
    parser.printMessage();

    cv::Ptr<cv::text::OCRTesseract> ocr = cv::text::OCRTesseract::create();
    std::string output;

    //Cargamos el clasificador
    if (!patenteIdentifier.load("haarcascade_russian_plate_number.xml"))
    {
        std::cout << "Error cargando el clasificador\n";
        return -1;
    }

    //Leemos la imagen que se paso por linea de comandos, si no se paso ninguna se toma test.jpg
    cv::Mat image = cv::imread(parser.get<std::string>("image_path"));

    //Convertimos a blanco y negro y ecualizamos el histograma
    cv::Mat imageGrayScale;
    cvtColor(image, imageGrayScale, cv::COLOR_BGR2GRAY);
    equalizeHist(imageGrayScale, imageGrayScale);

    //vector para las cajas de las patentes en la imagen
    std::vector<cv::Rect> patentes;

    //detectamos los revolveres
    patenteIdentifier.detectMultiScale(imageGrayScale, patentes);

    //Dibujamos los rectangulos que marcan las patentes
    for (size_t i = 0; i < patentes.size(); i++)
    {
        cv::rectangle(image, patentes[i], cv::Scalar(0, 0, 255), 4);
    }

    //Guardamos la imagen y la mostramos
    
    cv::imwrite("./output.png", image);
    cv::resize(image, image, cv::Size(), 1120.0 / image.cols,610.0 / image.rows);
    cv::imshow("Output", image);
    cv::Size size(1280, 720);
    if (patentes.size() > 0)
    {
        cv::Mat patente = imageGrayScale(patentes[0]);
        cv::resize(patente, patente, cv::Size(), 330.0 / patente.cols, 110.0 / patente.rows);
        cv::threshold(patente, patente, 100, 255, cv::THRESH_BINARY);
        cv::imshow("Patente", patente);
        ocr->run(patente, output);
    }
    

    std::cout << output << std::endl;

    cv::waitKey();

    return 0;


}