// ***************************************************************** -*- C++ -*-
// exifprint.cpp, $Rev: 3090 $
// Sample program to print the Exif metadata of an image
#include <exiv2/exiv2.hpp>
#include <iostream>
#include <iomanip>
#include <cassert>
int main(int argc, char* const argv[])
try {
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open("test.jpg");
    assert(image.get() != 0);
    image->readMetadata();
    Exiv2::ExifData& exifData = image->exifData();
    if (exifData.empty()) {
        std::string error;
        error += ": No Exif data found in the file";

    }
    Exiv2::ExifData::const_iterator end = exifData.end();
    for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i) {
        const char* tn = i->typeName();
        std::cout << std::setw(44) << std::setfill(' ') << std::left
            << i->key() << " "
            << "0x" << std::setw(4) << std::setfill('0') << std::right
            << std::hex << i->tag() << " "
            << std::setw(9) << std::setfill(' ') << std::left
            << (tn ? tn : "Unknown") << " "
            << std::dec << std::setw(3)
            << std::setfill(' ') << std::right
            << i->count() << "  "
            << std::dec << i->value()
            << "\n";
    }
    return 0;
}
//catch (std::exception& e) {
//catch (Exiv2::AnyError& e) {
catch (Exiv2::Error& e) {
    std::cout << "Caught Exiv2 exception '" << e.what() << "'\n";
    return -1;
}
