#include <iostream>
#include <iomanip>

#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"

int main()
{
    namespace fs = std::experimental::filesystem;

    const std::string inDir = "C:/Users/lee.namgoo/Repos/SuaKIT/SuaKIT/x64/Release/UnitTest/Detection/Single/sw_512x512/test_bk/";
    const std::string outDir = "C:/Users/lee.namgoo/Repos/SuaKIT/SuaKIT/x64/Release/UnitTest/Detection/Single/sw_512x512/test/";
    int count = 0;

    // experimental/filesystem 이용한 복사 & RapidJSON을 이용한 작업 example
    for (const auto& p : fs::directory_iterator(inDir))
    {
        const std::string fullPath = p.path().string();                                     // inDir 이하 각 directory_entry 들의 전체 패스
        const std::string grandparentPath = p.path().parent_path().parent_path().string();  // parent_path 연속해서 두 번 먹임
        const std::string filename = p.path().filename().string();                          // 파일 이름
        const std::string stem = p.path().stem().string();                                  // 확장자를 제외한 파일 이름
        const std::string extension = p.path().extension().string();                        // 확장자 ('.' 포함)

        if (extension == ".png") {                                                          // .png 파일은 단순 복사

            const size_t token = stem.find('_');
            const int mainNum = atoi(stem.substr(0, token).c_str());
            const int minorNum = atoi(stem.substr(token + 1).c_str());

            std::stringstream new_num_stream;
            new_num_stream << std::setw(3) << std::setfill('0') << mainNum << "_" << std::setw(3) << std::setfill('0') << minorNum << extension;

            const fs::path outPath(outDir + new_num_stream.str());

            // 복사!!
            fs::copy(p, outPath);

        } else if (extension == ".anno") {                                                  // .anno 파일은 JSON 포맷 바꿔서 .json으로 저장

            std::ifstream istream(fullPath);
            std::stringstream insstr;
            insstr << istream.rdbuf();

            rapidjson::Document inDoc, outDoc;

            // set outDoc
            outDoc.SetObject();
            rapidjson::Document::AllocatorType& allocator = outDoc.GetAllocator();
            outDoc.AddMember("__comment", "2.1.0", allocator);                              // JSON script 버전 정보

            // set inDoc
            inDoc.Parse(insstr.str().c_str());

            rapidjson::Value labels(rapidjson::kArrayType);
            rapidjson::Value masks(rapidjson::kArrayType);

            if (inDoc.HasMember("class1")) {
                const rapidjson::Value& class1 = inDoc["class1"];
                if (class1.IsArray()) {
                    for (auto& v : class1.GetArray()) {
                        rapidjson::Value label(rapidjson::kObjectType);
                        label.AddMember("Class", 0, allocator);
                        label.AddMember("Shape", "rectangle", allocator);
                        label.AddMember("StrokeThickness", 1.0f, allocator);

                        rapidjson::Value points(rapidjson::kArrayType);
                        {
                            std::string lt = std::to_string((float)v.GetArray()[0].GetInt()) + "," + std::to_string((float)v.GetArray()[1].GetInt());
                            std::string rb = std::to_string((float)v.GetArray()[2].GetInt()) + "," + std::to_string((float)v.GetArray()[3].GetInt());

                            rapidjson::Value lt_val;
                            rapidjson::Value rb_val;

                            lt_val.SetString(lt.c_str(), lt.length(), allocator);
                            rb_val.SetString(rb.c_str(), rb.length(), allocator);

                            points.PushBack(lt_val, allocator);
                            points.PushBack(rb_val, allocator);
                        }
                        label.AddMember("Points", points, allocator);

                        labels.PushBack(label, allocator);
                    }
                }
            }

            if (inDoc.HasMember("class2")) {
                const rapidjson::Value& class1 = inDoc["class2"];
                if (class1.IsArray()) {
                    for (auto& v : class1.GetArray()) {
                        rapidjson::Value label(rapidjson::kObjectType);
                        label.AddMember("Class", 1, allocator);
                        label.AddMember("Shape", "rectangle", allocator);
                        label.AddMember("StrokeThickness", 1.0f, allocator);

                        rapidjson::Value points(rapidjson::kArrayType);
                        {
                            std::string lt = std::to_string((float)v.GetArray()[0].GetInt()) + "," + std::to_string((float)v.GetArray()[1].GetInt());
                            std::string rb = std::to_string((float)v.GetArray()[2].GetInt()) + "," + std::to_string((float)v.GetArray()[3].GetInt());

                            rapidjson::Value lt_val;
                            rapidjson::Value rb_val;

                            lt_val.SetString(lt.c_str(), lt.length(), allocator);
                            rb_val.SetString(rb.c_str(), rb.length(), allocator);

                            points.PushBack(lt_val, allocator);
                            points.PushBack(rb_val, allocator);
                        }
                        label.AddMember("Points", points, allocator);

                        labels.PushBack(label, allocator);
                    }
                }
            }

            outDoc.AddMember("Labels", labels, allocator);
            outDoc.AddMember("Masks", masks, allocator);

            // 이제 출력 패스를 만든다.
            const size_t token = stem.find('_');
            const int mainNum = atoi(stem.substr(0, token).c_str());
            const int minorNum = atoi(stem.substr(token + 1).c_str());

            std::stringstream new_num_stream;
            new_num_stream << std::setw(3) << std::setfill('0') << mainNum << "_" << std::setw(3) << std::setfill('0') << minorNum << "_label.json";

            const fs::path outPath(outDir + new_num_stream.str());

            // 출력 패스에 JSON을 출력한다. rapidjson의 Stream을 이용한다.
            rapidjson::StringBuffer outBuffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(outBuffer);
            outDoc.Accept(writer);

            std::ofstream ostream(outPath);

            ostream.write(outBuffer.GetString(), outBuffer.GetSize());
        }
    }
}