

 var \_gaq = \_gaq || [];
 \_gaq.push(['\_setAccount', 'UA-17821189-2']);
 \_gaq.push(['\_setDomainName', 'wiki.ros.org']);
 \_gaq.push(['\_trackPageview']);

 (function() {
 var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
 ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
 var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
 })();

ROS/Tutorials/Auto - ROS Wiki

<!--
var search\_hint = "Search";
//-->

 window.dataLayer = window.dataLayer || [];
 function gtag(){dataLayer.push(arguments);}
 gtag('js', new Date());

 gtag('config', 'G-EVD5Z6G6NH');

<!--// Initialize search form
var f = document.getElementById('searchform');
if(f) f.getElementsByTagName('label')[0].style.display = 'none';
var e = document.getElementById('searchinput');
if(e) {
 searchChange(e);
 searchBlur(e);
}

function handleSubmit() {
 var f = document.getElementById('searchform');
 var t = document.getElementById('searchinput');
 var r = document.getElementById('real\_searchinput');

 //alert("handleSubmit "+ t.value);
 if(t.value.match(/review/)) {
 r.value = t.value;
 } else {
 //r.value = t.value + " -PackageReviewCategory -StackReviewCategory -M3Review -DocReview -ApiReview -HelpOn -BadContent -LocalSpellingWords";
 r.value = t.value + " -PackageReviewCategory -StackReviewCategory -DocReview -ApiReview";
 }
 //return validate(f);
}
//-->

|  |  |
| --- | --- |
| [ros.org](/ "/") | [About](http://www.ros.org/about-ros "http://www.ros.org/about-ros")
 |
 [Support](/Support "/Support")
 |
 [Discussion Forum](http://discourse.ros.org/ "http://discourse.ros.org/")
 |
 [Index](http://index.ros.org/ "http://index.ros.org/")
 |
 [Service Status](http://status.ros.org/ "http://status.ros.org/")
 |
 [Q&A answers.ros.org](http://answers.ros.org/ "http://answers.ros.org/") |
| [Documentation](/ "/")[Browse Software](https://index.ros.org/packages "https://index.ros.org/packages")[News](https://discourse.ros.org/c/general "https://discourse.ros.org/c/general")[Download](/ROS/Installation "/ROS/Installation") |

* [ROS](/ROS "/ROS")
* [Tutorials](/ROS/Tutorials "/ROS/Tutorials")
* [Auto](/ROS/Tutorials/Auto "/ROS/Tutorials/Auto")

#### ROS 2 Documentation

The ROS Wiki is for ROS 1. Are you using ROS 2 ([Humble](http://docs.ros.org/en/humble/ "http://docs.ros.org/en/humble/"), [Iron](http://docs.ros.org/en/iron/ "http://docs.ros.org/en/iron/"), or [Rolling](http://docs.ros.org/en/rolling/ "http://docs.ros.org/en/rolling/"))?   
[Check out the ROS 2 Project Documentation](http://docs.ros.org "http://docs.ros.org")  
Package specific documentation can be found on [index.ros.org](https://index.ros.org "https://index.ros.org")

# Wiki

* [Distributions](/Distributions "/Distributions")
* [ROS/Installation](/ROS/Installation "/ROS/Installation")
* [ROS/Tutorials](/ROS/Tutorials "/ROS/Tutorials")
* [RecentChanges](/RecentChanges "/RecentChanges")
* [ROS/Tutorials/Auto](/ROS/Tutorials/Auto "/ROS/Tutorials/Auto")

# Page

* Immutable Page
* [Comments](# "#")
* [Info](/action/info/ROS/Tutorials/Auto?action=info "/action/info/ROS/Tutorials/Auto?action=info")
* [Attachments](/action/AttachFile/ROS/Tutorials/Auto?action=AttachFile "/action/AttachFile/ROS/Tutorials/Auto?action=AttachFile")
* More Actions:

Raw Text
Print View
Render as Docbook
Delete Cache
------------------------
Check Spelling
Like Pages
Local Site Map
------------------------
Rename Page
Copy Page
Delete Page
------------------------
My Pages
Subscribe User
------------------------
Remove Spam
Revert to this revision
Package Pages
Sync Pages
------------------------
CreatePdfDocument
Load
RawFile
Save
SlideShow

<!--// Init menu
actionsMenuInit('More Actions:');
//-->

# User

* [Login](/action/login/ROS/Tutorials/Auto?action=login "/action/login/ROS/Tutorials/Auto?action=login")

## Beginner Level

1. [Understanding ROS Topics](/th/ROS/Tutorials/UnderstandingTopics "/th/ROS/Tutorials/UnderstandingTopics")This tutorial introduces ROS topics as well as using the [rostopic](/rostopic "/rostopic") and [rqt\_plot](/rqt_plot "/rqt_plot") commandline tools.
2. [Writing a Simple Service and Client (Python)](/th/ROS/Tutorials/WritingServiceClient%28python%29 "/th/ROS/Tutorials/WritingServiceClient%28python%29")This tutorial covers how to write a service and client node in python.
3. [安装和配置ROS环境](/cn/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/cn/ROS/Tutorials/InstallingandConfiguringROSEnvironment")本教程将指导您在计算机上安装ROS和配置ROS环境。
4. [การติดตั้ง และ ปรับตั้ง ROS Environment](/th/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/th/ROS/Tutorials/InstallingandConfiguringROSEnvironment")แบบเรียนนี้ เป็นการเรียนรู้เกี่ยวกับการติดตั้งระบบ ROS และ การปรับตั้ง ROS environment ในเครื่องคอมพิวเตอร์ของคุณ
5. [Navigating the ROS Filesystem](/th/ROS/Tutorials/NavigatingTheFilesystem "/th/ROS/Tutorials/NavigatingTheFilesystem")This tutorial introduces ROS filesystem concepts, and covers using the roscd, rosls, and [rospack](/rospack "/rospack") commandline tools.
6. [Instalando e Configurado o Seu Ambiente ROS](/pt_BR/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/pt_BR/ROS/Tutorials/InstallingandConfiguringROSEnvironment")Este tutorial te leva pelo processo de instalação do ROS e configuração do ambiente no seu computador.
7. [Comprendre les 'nodes' ROS](/fr/ROS/Tutorials/UnderstandingNodes "/fr/ROS/Tutorials/UnderstandingNodes")Ce tutoriel introduit les concepts de ROS graph et l'utilisation des outils en ligne de commande [roscore](/roscore "/roscore"), [rosnode](/rosnode "/rosnode"), et [rosrun](/rosrun "/rosrun").
8. [Comprendre les Topics ROS](/fr/ROS/Tutorials/UnderstandingTopics "/fr/ROS/Tutorials/UnderstandingTopics")Ce tutoriel introduit les concepts de Topics sous ROS ainsi que l'utilisation des outils en ligne de commande [rostopic](/rostopic "/rostopic") et [rqt\_plot](/rqt_plot "/rqt_plot").
9. [Общие сведения о узлах в ROS](/ru/ROS/Tutorials/UnderstandingNodes "/ru/ROS/Tutorials/UnderstandingNodes")Данное руководство знакомит с понятиями графа ROS и описывает использование инструментов командной строки: [roscore](/roscore "/roscore"), [rosnode](/rosnode "/rosnode"), и [rosrun](/rosrun "/rosrun").
10. [Writing Simple Service and Client (EusLisp)](/ROS/Tutorials/WritingServiceClient%28euslisp%29 "/ROS/Tutorials/WritingServiceClient%28euslisp%29")This tutorial covers how to write a service and client node in euslisp.
11. [Navigating the ROS wiki](/th/ROS/Tutorials/NavigatingTheWiki "/th/ROS/Tutorials/NavigatingTheWiki")This tutorial discusses the layout of the ROS wiki ([ros.org](/Documentation "/Documentation")) and talks about how to find what you want to know.
12. [Getting started with roswtf](/th/ROS/Tutorials/Getting%20started%20with%20roswtf "/th/ROS/Tutorials/Getting%20started%20with%20roswtf")Basic introduction to the [roswtf](/roswtf "/roswtf") tool.
13. [Using rxconsole and roslaunch](/th/ROS/Tutorials/UsingRxconsoleRoslaunch "/th/ROS/Tutorials/UsingRxconsoleRoslaunch")This tutorial introduces ROS using [rxconsole](/rxconsole "/rxconsole") and rxloggerlevel for debugging and [roslaunch](/roslaunch "/roslaunch") for starting many nodes at once.
14. [Cài đặt và cấu hình môi trường cho ROS](/vn/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/vn/ROS/Tutorials/InstallingandConfiguringROSEnvironment")Hướng dẫn cài đặt và cấu hình môi trường cho ROS trên máy tính.
15. [Recording and playing back data](/th/ROS/Tutorials/Recording%20and%20playing%20back%20data "/th/ROS/Tutorials/Recording%20and%20playing%20back%20data")This tutorial will teach you how to record data from a running ROS system into a .bag file, and then to play back the data to produce similar behavior in a running system.
16. [Writing a Simple Service and Client (C++)](/th/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29 "/th/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29")This tutorial covers how to write a service and client node in C++.
17. [Navegando pelo Sistema de Arquivos do ROS](/pt_BR/ROS/Tutorials/NavigatingTheFilesystem "/pt_BR/ROS/Tutorials/NavigatingTheFilesystem")Este tutorial introduz os conceitos de sistema de arquivos do ROS e cobre a utilização das ferramentas de linha de comando roscd, rosls, e [rospack](/rospack "/rospack").
18. [Criando um Pacote ROS](/pt_BR/ROS/Tutorials/CreatingPackage "/pt_BR/ROS/Tutorials/CreatingPackage")Este tutorial cobre a utilização de [roscreate-pkg](/roscreate "/roscreate") ou [catkin](/catkin "/catkin") para criar um novo pacote, e [rospack](/rospack "/rospack") para listar dependências de pacotes.
19. [تثبيت بيئة ROS وإعدادها](/ar/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/ar/ROS/Tutorials/InstallingandConfiguringROSEnvironment")يساعدك هذا الدرس في تثبيت ROS، وفي تجهيز بيئة ROS على حاسوبك.
20. [التعرف على نظام ملفات ROS](/ar/ROS/Tutorials/NavigatingTheFilesystem "/ar/ROS/Tutorials/NavigatingTheFilesystem")يقدم هذا الدرس مفاهيم نظام الملفات في ROS، ويتضمن استخدام أدوات موجه الأوامر roscd و rosls و [rospack](/rospack "/rospack").
21. [ROSのmsgとsrvを作る](/ja/ROS/Tutorials/CreatingMsgAndSrv "/ja/ROS/Tutorials/CreatingMsgAndSrv")このチュートリアルでは、どのようにmsgやsrvファイルを作りビルドするかを[rosmsg](/rosmsg "/rosmsg"), rossrv や roscpなどのコマンドの使い方とともに学びます
22. [シンプルな配信者(Publisher)と購読者(Subscriber)を書く(Python)](/ja/ROS/Tutorials/WritingPublisherSubscriber%28python%29 "/ja/ROS/Tutorials/WritingPublisherSubscriber%28python%29")このチュートリアルでは, 配信者(Publisher)と購読者(Subscriber)のPythonでの書き方について学びます
23. [Criando um Nó de Serviço-Cliente simples (C++)](/pt_BR/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29 "/pt_BR/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29")Neste tutorial é ensinado como escrever um Nó (node) de Serviço-Cliente em C++.
24. [ROSのノードを理解する](/ja/ROS/Tutorials/UnderstandingNodes "/ja/ROS/Tutorials/UnderstandingNodes")このチュートリアルではROSのグラフの概念を知り，[roscore](/roscore "/roscore")，[rosnode](/rosnode "/rosnode")と[rosrun](/rosrun "/rosrun")などのコマンドラインツールの使い方を学びます．
25. [ROS環境のインストールとセットアップ](/ja/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/ja/ROS/Tutorials/InstallingandConfiguringROSEnvironment")このチュートリアルでは, ROSのインストールと環境セットアップを行います．
26. [ROSパッケージを作る](/ja/ROS/Tutorials/CreatingPackage "/ja/ROS/Tutorials/CreatingPackage")このチュートリアルでは，[roscreate-pkg](/roscreate "/roscreate")やcatkin\_create\_pkgで新しいパッケージを作る方法やパッケージの依存関係を表示する[rospack](/rospack "/rospack")の使い方を学びます．
27. [Understanding ROS Services and Parameters](/th/ROS/Tutorials/UnderstandingServicesParams "/th/ROS/Tutorials/UnderstandingServicesParams")This tutorial introduces ROS services, and parameters as well as using the [rosservice](/rosservice "/rosservice") and [rosparam](/rosparam "/rosparam") commandline tools.
28. [Installing and Configuring Your ROS Environment](/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/ROS/Tutorials/InstallingandConfiguringROSEnvironment")This tutorial walks you through installing ROS and setting up the ROS environment on your computer.
29. [Writing a Simple Publisher and Subscriber (C++)](/th/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29 "/th/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29")This tutorial covers how to write a publisher and subscriber node in C++.
30. [データを記録し，リプレイをする](/ja/ROS/Tutorials/Recording%20and%20playing%20back%20data "/ja/ROS/Tutorials/Recording%20and%20playing%20back%20data")このチュートリアルでは，実行中の ROS のシステムから得られるデータをどのように .bag ファイルに保存し，どのように同じような状況を再現させるかを学習します．
31. [Using rosed to edit files in ROS](/th/ROS/Tutorials/UsingRosEd "/th/ROS/Tutorials/UsingRosEd")This tutorial shows how to use [rosed](/rosbash "/rosbash") to make editing easier.
32. [rqt\_console と roslaunch　を使う](/ja/ROS/Tutorials/UsingRxconsoleRoslaunch "/ja/ROS/Tutorials/UsingRxconsoleRoslaunch")このチュートリアルは, ROSでデバッグのために[rqt\_console](/rqt_console "/rqt_console")や[rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level")を使うことや, 一度に複数のnodeを起動する[roslaunch](/roslaunch "/roslaunch")の使うことを紹介します. ROS fuerte, もしくは, それ以前の[rqt](/rqt "/rqt")が完全な状態で提供されていないディストリビューションを使用している場合, 古いrxベースのツールを使用している[こちらのページ](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch")を参照してください.
33. [Comprendre les services et paramètres ROS](/fr/ROS/Tutorials/UnderstandingServicesParams "/fr/ROS/Tutorials/UnderstandingServicesParams")Ce tutoriel introduit les concepts de services et de paramètres sous ROS, ainsi que l'utilisation des outils en ligne de commande [rosservice](/rosservice "/rosservice") et [rosparam](/rosparam "/rosparam").
34. [Creating a ROS msg and srv](/th/ROS/Tutorials/CreatingMsgAndSrv "/th/ROS/Tutorials/CreatingMsgAndSrv")This tutorial covers how to create and build msg and srv files as well as the [rosmsg](/rosmsg "/rosmsg"), rossrv and roscp commandline tools.
35. [シンプルな配信者(Publisher)と購読者(Subscriber)を書く(C++)](/ja/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29 "/ja/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29")このチュートリアルでは, 配信者(Publisher)と購読者(Subscriber)のC++での書き方について学びます
36. [Установка и настройка рабочего окружения ROS](/ru/ROS/Tutorials/InstallingandConfiguringROSEnvironment "/ru/ROS/Tutorials/InstallingandConfiguringROSEnvironment")В этом учебном пособии вы узнаете, как установить и настроить ROS на вашем компьютере.
37. [シンプルなサービスとクライアントを書く (Python)](/ja/ROS/Tutorials/WritingServiceClient%28python%29 "/ja/ROS/Tutorials/WritingServiceClient%28python%29")このチュートリアルはPythonでのサービスとクライアントノードの書き方について扱います
38. [Using rqt\_console and roslaunch](/th/ROS/Tutorials/UsingRqtconsoleRoslaunch "/th/ROS/Tutorials/UsingRqtconsoleRoslaunch")This tutorial introduces ROS using [rqt\_console](/rqt_console "/rqt_console") and [rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level") for debugging and [roslaunch](/roslaunch "/roslaunch") for starting many nodes at once. If you use ROS fuerte or ealier distros where [rqt](/rqt "/rqt") isn't fully available, please see this page with [this page](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch") that uses old rx based tools.
39. [Writing a Simple Service and Client (C++)](/cn/ROS/Tutorials/WritingServiceClient "/cn/ROS/Tutorials/WritingServiceClient")This tutorial covers how to write a service and client node in C++.
40. [Navigating the ROS Filesystem](/ROS/Tutorials/NavigatingTheFilesystem "/ROS/Tutorials/NavigatingTheFilesystem")This tutorial introduces ROS filesystem concepts, and covers using the roscd, rosls, and [rospack](/rospack "/rospack") commandline tools.
41. [Writing a Simple Publisher and Subscriber (Python)](/th/ROS/Tutorials/WritingPublisherSubscriber%28python%29 "/th/ROS/Tutorials/WritingPublisherSubscriber%28python%29")This tutorial covers how to write a publisher and subscriber node in python.
42. [Examining the Simple Service and Client](/th/ROS/Tutorials/ExaminingServiceClient "/th/ROS/Tutorials/ExaminingServiceClient")This tutorial examines running the simple service and client.
43. [Using rxconsole and roslaunch](/pt_BR/ROS/Tutorials/UsingRxconsoleRoslaunch "/pt_BR/ROS/Tutorials/UsingRxconsoleRoslaunch")This tutorial introduces ROS using [rxconsole](/rxconsole "/rxconsole") and rxloggerlevel for debugging and [roslaunch](/roslaunch "/roslaunch") for starting many nodes at once.
44. [Tìm hiểu trình phục vụ và các tham số trong ROS](/vn/ROS/Tutorials/UnderstandingServicesParams "/vn/ROS/Tutorials/UnderstandingServicesParams")Hướng dẫn này giới thiệu các dịch vụ ROS, và các tham số cũng như sử dụng các công cụ dòng lệnh rosservice và rosparam.
45. [Dùng rqt\_console và roslaunch](/vn/ROS/Tutorials/UsingRqtconsoleRoslaunch "/vn/ROS/Tutorials/UsingRqtconsoleRoslaunch")Hướng dẫn dùng ROS [rqt\_console](/rqt_console "/rqt_console") và [rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level") để gỡ rối và [roslaunch](/roslaunch "/roslaunch") để khởi chạy nhiều nút một lúc. Nếu bạn dùng ROS fuerte hoặc bản cũ hơn thì [rqt](/rqt "/rqt") không có sẵn, xem trang này với [this page](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch") dùng rx cũ hơn .
46. [Escrevendo um Serviço-Cliente simples (Python)](/pt_BR/ROS/Tutorials/WritingServiceClient%28python%29 "/pt_BR/ROS/Tutorials/WritingServiceClient%28python%29")Esse tutorial cobre como escrever um nó serviço-cliente simples em python.
47. [Examinando um Serviço-Cliente Simples](/pt_BR/ROS/Tutorials/ExaminingServiceClient "/pt_BR/ROS/Tutorials/ExaminingServiceClient")Este tutorial consiste em mostrar como é a interação entre um serviço e um cliente através da adição de dois números.
48. [Navigating the ROS Filesystem](/ru/ROS/Tutorials/NavigatingTheFilesystem "/ru/ROS/Tutorials/NavigatingTheFilesystem")This tutorial introduces ROS filesystem concepts, and covers using the roscd, rosls, and [rospack](/rospack "/rospack") commandline tools.
49. [Creating a ROS Package](/ROS/Tutorials/CreatingPackage "/ROS/Tutorials/CreatingPackage")This tutorial covers using [roscreate-pkg](/roscreate "/roscreate") or [catkin](/catkin "/catkin") to create a new package, and [rospack](/rospack "/rospack") to list package dependencies.
50. [Gerando um pacote no ROS](/pt_BR/ROS/Tutorials/BuildingPackages "/pt_BR/ROS/Tutorials/BuildingPackages")Este tutorial cobre as ferramentas e métodos para a geração de um pacote no ROS.
51. [Um pouco mais sobre nós(''nodes'') no ROS](/pt_BR/ROS/Tutorials/UnderstandingNodes "/pt_BR/ROS/Tutorials/UnderstandingNodes")Este tutorial apresenta os conceitos introdutórios sobre os componentes básicos do ROS (também referenciados como ROS Graph) e discute o uso das ferramentas utilizadas via terminal: [roscore](/roscore "/roscore"), [rosnode](/rosnode "/rosnode") e [rosrun](/rosrun "/rosrun").
52. [Entendendo tópicos no ROS](/pt_BR/ROS/Tutorials/UnderstandingTopics "/pt_BR/ROS/Tutorials/UnderstandingTopics")Este tutorial apresenta tópicos no ROS, e também como utilizar as ferramentas [rostopic](/rostopic "/rostopic") e [rqt\_plot](/rqt_plot "/rqt_plot").
53. [Entendendo Serviços e Parâmetros ROS](/pt_BR/ROS/Tutorials/UnderstandingServicesParams "/pt_BR/ROS/Tutorials/UnderstandingServicesParams")Esse tutorial introduz Serviços e Parâmetros ROS, bem como o uso das ferramentas de linha de comando [rosservice](/rosservice "/rosservice") e [rosparam](/rosparam "/rosparam").
54. [Usando o rqt\_console e o roslaunch](/pt_BR/ROS/Tutorials/UsingRqtconsoleRoslaunch "/pt_BR/ROS/Tutorials/UsingRqtconsoleRoslaunch")Esse tutorial introduces ROS using [rqt\_console](/rqt_console "/rqt_console") and [rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level") for debugging and [roslaunch](/roslaunch "/roslaunch") for starting many nodes at once. If you use ROS fuerte or ealier distros where [rqt](/rqt "/rqt") isn't fully available, please see this page with [this page](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch") that uses old rx based tools.
55. [Utilizando o rosed para edição de arquivos no ROS.](/pt_BR/ROS/Tutorials/UsingRosEd "/pt_BR/ROS/Tutorials/UsingRosEd")Este tutorial mostra como usar o [rosed](/rosbash "/rosbash") para facilitar a edição de arquivos.
56. [Criando arquivos msg e srv no ROS](/pt_BR/ROS/Tutorials/CreatingMsgAndSrv "/pt_BR/ROS/Tutorials/CreatingMsgAndSrv")Este tutorial ensina como criar e construir (build) arquivos .msg e .srv assim como as linhas de comando de atalho: [rosmsg](/rosmsg "/rosmsg"), [rossrv](/rossrv "/rossrv") and [roscp](/roscp "/roscp").
57. [Escrevendo um Simples Publicador(Publisher) e Subscrito (Subscriber) (Python)](/pt_BR/ROS/Tutorials/WritingPublisherSubscriber%28python%29 "/pt_BR/ROS/Tutorials/WritingPublisherSubscriber%28python%29")Esse tutorial explica como escrever um nó publicador (publisher) e subscrito (subscriber) em python.
58. [Criando um simple Publicador e Subscritor (C++)](/pt_BR/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29 "/pt_BR/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29")Este tutorial This tutorial apresenta como criar um nó publicador e um nó subscritor em C++.
59. [Using rxconsole and roslaunch](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch")This tutorial introduces ROS using [rxconsole](/rxconsole "/rxconsole") and rxloggerlevel for debugging and [roslaunch](/roslaunch "/roslaunch") for starting many nodes at once.
60. [Understanding ROS Nodes](/th/ROS/Tutorials/UnderstandingNodes "/th/ROS/Tutorials/UnderstandingNodes")This tutorial introduces ROS graph concepts and discusses the use of [roscore](/roscore "/roscore"), [rosnode](/rosnode "/rosnode"), and [rosrun](/rosrun "/rosrun") commandline tools.
61. [ROSトピックの理解](/ja/ROS/Tutorials/UnderstandingTopics "/ja/ROS/Tutorials/UnderstandingTopics")このチュートリアルは[rostopic](/rostopic "/rostopic") や [rqt\_plot](/rqt_plot "/rqt_plot")などのコマンドとともに，ROSのtopicについて学びます．
62. [ROSのサービスとパラメータを理解する](/ja/ROS/Tutorials/UnderstandingServicesParams "/ja/ROS/Tutorials/UnderstandingServicesParams")このチュートリアルでは、[rosservice](/rosservice "/rosservice")や [rosparam](/rosparam "/rosparam")などのコマンドラインツールとともに、ROSのサービスやパラメータなどについて学びます
63. [rqt\_console と roslaunch を使う](/ja/ROS/Tutorials/UsingRqtconsoleRoslaunch "/ja/ROS/Tutorials/UsingRqtconsoleRoslaunch")このチュートリアルでは, ROSのデバッグで使う[rqt\_console](/rqt_console "/rqt_console")や[rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level")と, 一度に複数のnodeを起動する[roslaunch](/roslaunch "/roslaunch")の使い方を学びます。ROS fuerte, もしくは, それ以前の[rqt](/rqt "/rqt")が完全な状態で提供されていないディストリビューションを使用している場合, 古いrxベースのツールを使用している[こちらのページ](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch")を参照してください.
64. [Writing Simple Service and Client (EusLisp)](/pt_BR/ROS/Tutorials/WritingServiceClient%28euslisp%29 "/pt_BR/ROS/Tutorials/WritingServiceClient%28euslisp%29")This tutorial covers how to write a service and client node in euslisp.
65. [إنشاء حزمة ROS](/ar/ROS/Tutorials/CreatingPackage "/ar/ROS/Tutorials/CreatingPackage")يشمل هذا الدرس [roscreate-pkg](/roscreate "/roscreate") و [catkin](/catkin "/catkin") لإنشاء حزمة جديدة، و [rospack](/rospack "/rospack") لعرض قائمة بأسماء المكتبات التي تعتمد عليها الحزمة.
66. [ROS文件系统导览](/cn/ROS/Tutorials/NavigatingTheFilesystem "/cn/ROS/Tutorials/NavigatingTheFilesystem")本教程介绍ROS文件系统的概念，包括如何使用roscd、rosls和[rospack](/rospack "/rospack")命令行工具。
67. [创建ROS软件包](/cn/ROS/Tutorials/CreatingPackage "/cn/ROS/Tutorials/CreatingPackage")本教程介绍如何使用[roscreate-pkg](/roscreate "/roscreate")或[catkin](/catkin "/catkin")创建新的ROS软件包，并使用[rospack](/rospack "/rospack")列出软件包的依赖关系。
68. [构建ROS软件包](/cn/ROS/Tutorials/BuildingPackages "/cn/ROS/Tutorials/BuildingPackages")本教程介绍了构建软件包及使用的工具链。
69. [理解ROS节点](/cn/ROS/Tutorials/UnderstandingNodes "/cn/ROS/Tutorials/UnderstandingNodes")该教程介绍了ROS[图](/cn/ROS/Concepts#ROS.2Bi6F7l1b.2BXEJrIQ- "/cn/ROS/Concepts#ROS.2Bi6F7l1b.2BXEJrIQ-")的概念，并探讨了[roscore](/roscore "/roscore")、[rosnode](/rosnode "/rosnode")和[rosrun](/rosrun "/rosrun")命令行工具的使用。
70. [理解ROS话题](/cn/ROS/Tutorials/UnderstandingTopics "/cn/ROS/Tutorials/UnderstandingTopics")本教程介绍了ROS话题，以及如何使用[rostopic](/rostopic "/rostopic")和[rqt\_plot](/rqt_plot "/rqt_plot")命令行工具。
71. [理解ROS服务和参数](/cn/ROS/Tutorials/UnderstandingServicesParams "/cn/ROS/Tutorials/UnderstandingServicesParams")本教程介绍了ROS服务和参数的知识，以及命令行工具[rosservice](/rosservice "/rosservice")和[rosparam](/rosparam "/rosparam")的使用方法。
72. [使用rqt\_console和roslaunch](/cn/ROS/Tutorials/UsingRqtconsoleRoslaunch "/cn/ROS/Tutorials/UsingRqtconsoleRoslaunch")本教程介绍在ROS中使用[rqt\_console](/rqt_console "/rqt_console")和[rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level")进行调试，以及使用[roslaunch](/roslaunch "/roslaunch")同时启动多个节点。
73. [使用rosed在ROS中编辑文件](/cn/ROS/Tutorials/UsingRosEd "/cn/ROS/Tutorials/UsingRosEd")本教程展示了如何使用[rosed](/rosbash "/rosbash")来简化编辑过程。
74. [创建ROS消息和服务](/cn/ROS/Tutorials/CreatingMsgAndSrv "/cn/ROS/Tutorials/CreatingMsgAndSrv")本教程介绍如何创建和构建msg和srv文件，以及[rosmsg](/rosmsg "/rosmsg")、rossrv和roscp命令行工具的使用。
75. [编写简单的发布者和订阅者（C++）](/cn/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29 "/cn/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29")本教程介绍如何用C++编写发布者和订阅者节点。
76. [编写简单的发布者和订阅者（Python）](/cn/ROS/Tutorials/WritingPublisherSubscriber%28python%29 "/cn/ROS/Tutorials/WritingPublisherSubscriber%28python%29")本教程介绍如何用Python编写发布者和订阅者节点。
77. [检验简单的发布者和订阅者](/cn/ROS/Tutorials/ExaminingPublisherSubscriber "/cn/ROS/Tutorials/ExaminingPublisherSubscriber")本教程将介绍如何运行及测试发布者和订阅者。
78. [编写简单的服务和客户端（Python）](/cn/ROS/Tutorials/WritingServiceClient%28python%29 "/cn/ROS/Tutorials/WritingServiceClient%28python%29")本教程介绍如何用Python编写服务和客户端节点。
79. [编写简单的服务和客户端（C++）](/cn/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29 "/cn/ROS/Tutorials/WritingServiceClient%28c%2B%2B%29")本教程介绍如何用C++编写服务和客户端节点。
80. [检验简单的服务和客户端](/cn/ROS/Tutorials/ExaminingServiceClient "/cn/ROS/Tutorials/ExaminingServiceClient")本教程将介绍如何运行及测试服务和客户端。
81. [录制和回放数据](/cn/ROS/Tutorials/Recording%20and%20playing%20back%20data "/cn/ROS/Tutorials/Recording%20and%20playing%20back%20data")教你如何将正在运行的ROS系统中的数据记录到一个bag文件中，然后通过回放这些数据来来重现相似的运行过程。
82. [从bag文件中读取消息](/cn/ROS/Tutorials/reading%20msgs%20from%20a%20bag%20file "/cn/ROS/Tutorials/reading%20msgs%20from%20a%20bag%20file")了解从bag文件中读取所需话题的消息的两种方法，以及ros\_readbagfile脚本的使用。
83. [roswtf入门](/cn/ROS/Tutorials/Getting%20started%20with%20roswtf "/cn/ROS/Tutorials/Getting%20started%20with%20roswtf")简单介绍了[roswtf](/roswtf "/roswtf")工具的基本使用方法。
84. [探索ROS维基](/cn/ROS/Tutorials/NavigatingTheWiki "/cn/ROS/Tutorials/NavigatingTheWiki")本教程介绍了ROS维基([wiki.ros.org](/Documentation "/Documentation"))的组织结构以及使用方法。同时讲解了如何才能从ROS维基中找到你需要的信息。
85. [接下来做什么？](/cn/ROS/Tutorials/WhereNext "/cn/ROS/Tutorials/WhereNext")本教程将讨论获取更多知识的途径，以帮助你更好地使用ROS搭建真实或虚拟机器人。
86. [Creating a ROS Package](/th/ROS/Tutorials/CreatingPackage "/th/ROS/Tutorials/CreatingPackage")This tutorial covers using [roscreate-pkg](/roscreate "/roscreate") or [catkin](/catkin "/catkin") to create a new package, and [rospack](/rospack "/rospack") to list package dependencies.
87. [Building a ROS Package](/ROS/Tutorials/BuildingPackages "/ROS/Tutorials/BuildingPackages")This tutorial covers the toolchain to build a package.
88. [Dùng rosed để chỉnh sửa tập tin trong ROS](/vn/ROS/Tutorials/UsingRosEd "/vn/ROS/Tutorials/UsingRosEd")Hướng dẫn dùng [rosed](/rosbash "/rosbash") để dễ dàng chỉnh sửa.
89. [Tạo một ROS msg và srv](/vn/ROS/Tutorials/CreatingMsgAndSrv "/vn/ROS/Tutorials/CreatingMsgAndSrv")Hướng dẫn làm thế nào để tạo và xây dựng tập tin msg và srv [rosmsg](/rosmsg "/rosmsg"), công cụ dòng lệnh rossrv và roscp.
90. [Viết một Publisher và Subscriber (C++) đơn giản](/vn/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29 "/vn/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29")Hướng dẫn để viết một nút publisher và subscriber trong C++.
91. [Building a ROS Package](/th/ROS/Tutorials/BuildingPackages "/th/ROS/Tutorials/BuildingPackages")This tutorial covers the toolchain to build a package.
92. [Understanding ROS Nodes](/ROS/Tutorials/UnderstandingNodes "/ROS/Tutorials/UnderstandingNodes")This tutorial introduces ROS graph concepts and discusses the use of [roscore](/roscore "/roscore"), [rosnode](/rosnode "/rosnode"), and [rosrun](/rosrun "/rosrun") commandline tools.
93. [Understanding ROS Topics](/ROS/Tutorials/UnderstandingTopics "/ROS/Tutorials/UnderstandingTopics")This tutorial introduces ROS topics as well as using the [rostopic](/rostopic "/rostopic") and [rqt\_plot](/rqt_plot "/rqt_plot") commandline tools.
94. [Understanding ROS Services and Parameters](/ROS/Tutorials/UnderstandingServicesParams "/ROS/Tutorials/UnderstandingServicesParams")This tutorial introduces ROS services, and parameters as well as using the [rosservice](/rosservice "/rosservice") and [rosparam](/rosparam "/rosparam") commandline tools.
95. [Using rqt\_console and roslaunch](/ROS/Tutorials/UsingRqtconsoleRoslaunch "/ROS/Tutorials/UsingRqtconsoleRoslaunch")This tutorial introduces ROS using [rqt\_console](/rqt_console "/rqt_console") and [rqt\_logger\_level](/rqt_logger_level "/rqt_logger_level") for debugging and [roslaunch](/roslaunch "/roslaunch") for starting many nodes at once. If you use ROS fuerte or ealier distros where [rqt](/rqt "/rqt") isn't fully available, please see this page with [this page](/ROS/Tutorials/UsingRxconsoleRoslaunch "/ROS/Tutorials/UsingRxconsoleRoslaunch") that uses old rx based tools.
96. [Using rosed to edit files in ROS](/ROS/Tutorials/UsingRosEd "/ROS/Tutorials/UsingRosEd")This tutorial shows how to use [rosed](/rosbash "/rosbash") to make editing easier.
97. [Creating a ROS msg and srv](/ROS/Tutorials/CreatingMsgAndSrv "/ROS/Tutorials/CreatingMsgAndSrv")This tutorial covers how to create and build msg and srv files as well as the [rosmsg](/rosmsg "/rosmsg"), rossrv and roscp commandline tools.
98. [Writing a Simple Publisher and Subscriber (Python)](/ROS/Tutorials/WritingPublisherSubscriber%28python%29 "/ROS/Tutorials/WritingPublisherSubscriber%28python%29")This tutorial covers how to write a publisher and subscriber node in python.
99. [Writing a Simple Publisher and Subscriber (C++)](/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29 "/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29")This tutorial covers how to write a publisher and subscriber node in C++.
100. [Examining the Simple Publisher and Subscriber](/ROS/Tutorials/ExaminingPublisherSubscriber "/ROS/Tutorials/ExaminingPublisherSubscriber")This tutorial examines running the simple publisher and subscriber.
101. [Examining the Simple Publisher and Subscriber](/th/ROS/Tutorials/ExaminingPublisherSubscriber "/th/ROS/Tutorials/ExaminingPublisherSubscriber")This tutorial examines running the simple publisher and subscriber.
102. [roswtfを使う](/ja/ROS/Tutorials/Getting%20started%20with%20roswtf "/ja/ROS/Tutorials/Getting%20started%20with%20roswtf")このチュートリアルは、roswtfの基本的な使い方の導入です
103. [Hệ thống tập tin trong ROS](/vn/ROS/Tutorials/NavigatingTheFilesystem "/vn/ROS/Tutorials/NavigatingTheFilesystem")Khái niệm về tập tin hệ thống trong ROS, dùng các công cụ dòng lệnh (commandline) roscd, rosls, và [rospack](/rospack "/rospack").
104. [Tạo một gói ROS](/vn/ROS/Tutorials/CreatingPackage "/vn/ROS/Tutorials/CreatingPackage")Hướng dẫn tạo ra một gói ROS mới dùng [roscreate-pkg](/roscreate "/roscreate") hoặc [catkin](/catkin "/catkin") , và [rospack](/rospack "/rospack") để liệt kê danh sách gói phụ thuộc.
105. [Xây dựng một gói ROS](/vn/ROS/Tutorials/BuildingPackages "/vn/ROS/Tutorials/BuildingPackages")Hướng dẫn các công cụ xây dựng gói.
106. [Tìm hiểu về nút ROS](/vn/ROS/Tutorials/UnderstandingNodes "/vn/ROS/Tutorials/UnderstandingNodes")Hướng dẫn giới thiệu về khái niệm đồ thị trong ROS và thảo luận dùng công cụ dòng lệnh [roscore](/roscore "/roscore"), [rosnode](/rosnode "/rosnode"), and [rosrun](/rosrun "/rosrun").
107. [Gravando dados em tempo de execução e os reproduzindo posteriormente](/pt_BR/ROS/Tutorials/Recording%20and%20playing%20back%20data "/pt_BR/ROS/Tutorials/Recording%20and%20playing%20back%20data")Este tutorial possui o objetivo de orientá-lo na gravação dos dados gerados durante a execução de uma aplicação ROS em arquivos .bag. Assim como, na reprodução posterior destes dados com o intuito de representar o estado do sistema no momento da gravação.
108. [Introdução ao roswtf](/pt_BR/ROS/Tutorials/Getting%20started%20with%20roswtf "/pt_BR/ROS/Tutorials/Getting%20started%20with%20roswtf")Introdução à ferramenta [roswtf](/roswtf "/roswtf").
109. [Navegando na wiki do ROS](/pt_BR/ROS/Tutorials/NavigatingTheWiki "/pt_BR/ROS/Tutorials/NavigatingTheWiki")Este tutorial mostra como entender a estutura do wiki do ROS ([ros.org](/Documentation "/Documentation")) and talks about how to find what you want to know.
110. [Mais informações](/pt_BR/ROS/Tutorials/WhereNext "/pt_BR/ROS/Tutorials/WhereNext")Este tutorial discute opções para obter mais informações sobre o ROS em robôs reais e simulados.
111. [Viết một Publisher và Subscriber (Python) đơn giản](/vn/ROS/Tutorials/WritingPublisherSubscriber%28python%29 "/vn/ROS/Tutorials/WritingPublisherSubscriber%28python%29")Hướng dẫn để viết một nút publisher và subscriber trong python.
112. [Thực thi Publisher và Subscriber](/vn/ROS/Tutorials/ExaminingPublisherSubscriber "/vn/ROS/Tutorials/ExaminingPublisherSubscriber")Hướng dẫn chạy publisher và subscriber.
113. [Writing a Simple Service and Client (Python)](/ROS/Tutorials/WritingServiceClient%28python%29 "/ROS/Tutorials/WritingServiceClient%28python%29")This tutorial covers how to write a service and client node in python.
114. [Examining the Simple Service and Client](/ROS/Tutorials/ExaminingServiceClient "/ROS/Tutorials/ExaminingServiceClient")This tutorial examines running the simple service and client.
115. [Recording and playing back data](/ROS/Tutorials/Recording%20and%20playing%20back%20data "/ROS/Tutorials/Recording%20and%20playing%20back%20data")This tutorial will teach you how to record data from a running ROS system into a .bag file, and then to play back the data to produce similar behavior in a running system
116. [Reading messages from a bag file](/ROS/Tutorials/reading%20msgs%20from%20a%20bag%20file "/ROS/Tutorials/reading%20msgs%20from%20a%20bag%20file")Learn two ways to read messages from desired topics in a bag file, including using the ros\_readbagfile script.
117. [Getting started with roswtf](/ROS/Tutorials/Getting%20started%20with%20roswtf "/ROS/Tutorials/Getting%20started%20with%20roswtf")Basic introduction to the [roswtf](/roswtf "/roswtf") tool.
118. [Navigating the ROS wiki](/ROS/Tutorials/NavigatingTheWiki "/ROS/Tutorials/NavigatingTheWiki")This tutorial discusses the layout of the ROS wiki ([wiki.ros.org](/Documentation "/Documentation")) and talks about how to find what you want to know.
119. [Where Next?](/ROS/Tutorials/WhereNext "/ROS/Tutorials/WhereNext")This tutorial discusses options for getting to know more about using ROS on real or simulated robots.
120. [この先について。](/ja/ROS/Tutorials/WhereNext "/ja/ROS/Tutorials/WhereNext")このチュートリアルでは、シミュレーションや実機でROSについてもっと知るためのオプションについて説明します。
121. [Writing Simple Service and Client (EusLisp)](/ja/ROS/Tutorials/WritingServiceClient%28euslisp%29 "/ja/ROS/Tutorials/WritingServiceClient%28euslisp%29")This tutorial covers how to write a service and client node in [EusLisp](/EusLisp "/EusLisp").
122. [Where Next?](/th/ROS/Tutorials/WhereNext "/th/ROS/Tutorials/WhereNext")This tutorial discusses options for getting to know more about using ROS on real or simulated robots.

 **Now that you have completed the beginner level tutorials please answer this short [questionnaire](http://spreadsheets.google.com/viewform?formkey=dGJVOVhyXzd0b0YxRHAxWDdIZmo4cGc6MA "http://spreadsheets.google.com/viewform?formkey=dGJVOVhyXzd0b0YxRHAxWDdIZmo4cGc6MA").** 

---

## Intermediate Level

* More client API tutorials can be found in the relevant package ([roscpp](/roscpp/Tutorials "/roscpp/Tutorials"), [rospy](/rospy/Tutorials "/rospy/Tutorials"), [roslisp](/roslisp/Tutorials "/roslisp/Tutorials"))

1. [Creating a ROS package by hand.](/th/ROS/Tutorials/Creating%20a%20Package%20by%20Hand "/th/ROS/Tutorials/Creating%20a%20Package%20by%20Hand")This tutorial explains how to manually create a ROS package.
2. [手动创建ROS package](/cn/ROS/Tutorials/Creating%20a%20Package%20by%20Hand "/cn/ROS/Tutorials/Creating%20a%20Package%20by%20Hand")本教程将展示如何手动创建ROS package
3. [大きなプロジェクトのためのRoslaunch tips](/ja/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects "/ja/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects")このチュートリアルでは、大きなプロジェクトのためにroslaunchファイルを書く際のtipsについて説明しています。主に、さまざまなほかの状況にもできるだけ再利用できようにlaunchファイルを構成することを重点をおきます。ここではケーススタディに2dnav\_pr2を用います。
4. [Roslaunch在大型项目中的使用技巧](/cn/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects "/cn/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects")本教程主要介绍roslaunch在大型项目中的使用技巧。重点关注如何构建launch文件使得它能够在不同的情况下重复利用。我们将使用 2dnav\_pr2 package作为学习案例。
5. [Roslaunch tips for large projects](/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects "/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects")This tutorial describes some tips for writing roslaunch files for large projects. The focus is on how to structure launch files so they may be reused as much as possible in different situations. We'll use the 2dnav\_pr2 package as a case study.
6. [Running ROS across multiple machines](/th/ROS/Tutorials/MultipleMachines "/th/ROS/Tutorials/MultipleMachines")This tutorial explains how to start a ROS system using two machines. It explains the use of ROS\_MASTER\_URI to configure multiple machines to use a single master.
7. [Dicas de Roslaunch para projetos grandes](/pt_BR/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects "/pt_BR/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects")Traz dicas para escrever arquivos *roslaunch* para grandes projetos. O foco é em como estruturar os arquivos para que eles possam ser reutilizados o máximo possível em situações diversas. Usaremos o pacote *2dnav\_pr2* como caso de estudo.
8. [Roslaunch tips for large projects](/th/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects "/th/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects")This tutorial describes some tips for writing roslaunch files for large projects. The focus is on how to structure launch files so they may be reused as much as possible in different situations. We'll use the 2dnav\_pr2 package as a case study.
9. [複数のマシン上でROSを実行する](/ja/ROS/Tutorials/MultipleMachines "/ja/ROS/Tutorials/MultipleMachines")このチュートリアルは、どのように２つのマシンでROSのシステムを起動するかを説明します。つまり、ひとつのマスターで複数のマシンを管理するためのROS\_MASTER\_URIの使い方を説明します。
10. [Creating a ROS package by hand.](/ROS/Tutorials/Creating%20a%20Package%20by%20Hand "/ROS/Tutorials/Creating%20a%20Package%20by%20Hand")This tutorial explains how to manually create a ROS package.
11. [Creating a ROS package by hand.](/ko/ROS/Tutorials/Creating%20a%20Package%20by%20Hand "/ko/ROS/Tutorials/Creating%20a%20Package%20by%20Hand")This tutorial explains how to manually create a ROS package.
12. [Rodando o ROS em múltiplas máquinas.](/pt_BR/ROS/Tutorials/MultipleMachines "/pt_BR/ROS/Tutorials/MultipleMachines")Este tutorial explica como iniciar um sistema ROS usando duas máquinas. Ele explica como usar o ROS\_MASTER\_URI para confgirar múltiplas máquinas para usar um master único.
13. [Definindo mensagens customizadas](/pt_BR/ROS/Tutorials/DefiningCustomMessages "/pt_BR/ROS/Tutorials/DefiningCustomMessages")Este tutorial vai mostrar como definir uma mensagem customizada usando os tipos de mensagens disponíveis no ROS [Message Description Language](/pt_BR/ROS/Message_Description_Language "/pt_BR/ROS/Message_Description_Language").
14. [Defining Custom Messages](/th/ROS/Tutorials/DefiningCustomMessages "/th/ROS/Tutorials/DefiningCustomMessages")This tutorial will show you how to define your own custom message data types using the ROS [Message Description Language](/ROS/Message_Description_Language "/ROS/Message_Description_Language").
15. [Running ROS across multiple machines](/ROS/Tutorials/MultipleMachines "/ROS/Tutorials/MultipleMachines")This tutorial explains how to start a ROS system using two machines. It explains the use of ROS\_MASTER\_URI to configure multiple machines to use a single master.
16. [Criando um pacote de ROS manualmente](/pt_BR/ROS/Tutorials/Creating%20a%20Package%20by%20Hand "/pt_BR/ROS/Tutorials/Creating%20a%20Package%20by%20Hand")Como criar um pacote do ROS sem usar ferramentas.
17. [AppArmor and ROS](/SROS/Tutorials/AppArmorAndROS "/SROS/Tutorials/AppArmorAndROS")This tutorial explains how [AppArmor](/AppArmor "/AppArmor") can be used with ROS.
18. [Installing AppArmor Profiles for ROS](/SROS/Tutorials/InstallingAppArmorProfilesForROS "/SROS/Tutorials/InstallingAppArmorProfilesForROS")This tutorial explains how to install [AppArmor](/AppArmor "/AppArmor") Profiles to be used for securing ROS.
19. [Customizing AppArmor Profiles for ROS](/SROS/Tutorials/CustomizingAppArmorProfilesForROS "/SROS/Tutorials/CustomizingAppArmorProfilesForROS")This tutorial explains how to customizing [AppArmor](/AppArmor "/AppArmor") Profiles to be used for securing ROS.
20. [Roslaunch tips for large projects](/ko/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects "/ko/ROS/Tutorials/Roslaunch%20tips%20for%20larger%20projects")This tutorial describes some tips for writing roslaunch files for large projects. The focus is on how to structure launch files so they may be reused as much as possible in different situations. We'll use the 2dnav\_pr2 package as a case study.
21. [Running ROS across multiple machines](/ko/ROS/Tutorials/MultipleMachines "/ko/ROS/Tutorials/MultipleMachines")This tutorial explains how to start a ROS system using two machines. It explains the use of ROS\_MASTER\_URI to configure multiple machines to use a single master.
22. [Defining Custom Messages](/ko/ROS/Tutorials/DefiningCustomMessages "/ko/ROS/Tutorials/DefiningCustomMessages")This tutorial will show you how to define your own custom message data types using the ROS [Message Description Language](/ROS/Message_Description_Language "/ROS/Message_Description_Language").
23. [ROSのパッケージを手動で作る](/ja/ROS/Tutorials/Creating%20a%20Package%20by%20Hand "/ja/ROS/Tutorials/Creating%20a%20Package%20by%20Hand")このチュートリアルは、どのようにROSのパッケージを手動で作るかを説明します。
24. [ROS在多机器人上的使用](/cn/ROS/Tutorials/MultipleMachines "/cn/ROS/Tutorials/MultipleMachines")本教程将展示如何在两台机器上使用ROS系统，详述了使用ROS\_MASTER\_URI来配置多台机器使用同一个master。
25. [カスタムメッセージを定義する](/ja/ROS/Tutorials/DefiningCustomMessages "/ja/ROS/Tutorials/DefiningCustomMessages")このチュートリアルでは、ROS[Message Description Language](/ROS/Message_Description_Language "/ROS/Message_Description_Language")を用いて独自のカスタムメッセージを定義する方法をお見せします。
26. [Defining Custom Messages](/ROS/Tutorials/DefiningCustomMessages "/ROS/Tutorials/DefiningCustomMessages")This tutorial will show you how to define your own custom message data types using the ROS [Message Description Language](/ROS/Message_Description_Language "/ROS/Message_Description_Language").
27. [Transport Security and ROS](/SROS/Tutorials/TrasportSecurityAndROS "/SROS/Tutorials/TrasportSecurityAndROS")This tutorial explains what, why, and how TLS is used in SROS.
28. [Keyserver and SROS](/SROS/Tutorials/KeyserverAndSROS "/SROS/Tutorials/KeyserverAndSROS")This tutorial explains what, why, and how the keyserver is used in SROS.
29. [Running Keyserver](/SROS/Tutorials/RunningKeyserver "/SROS/Tutorials/RunningKeyserver")This tutorial explains how to run the keyserver.
30. [No Title](/th/ROS/Tutorials/Auto "/th/ROS/Tutorials/Auto")No Description
31. [Installing ROS Indigo in a chroot](/pt_BR/ROS/Tutorials/InstallingIndigoInChroot "/pt_BR/ROS/Tutorials/InstallingIndigoInChroot")This tutorial walks you through installing ROS Indigo (and Ubuntu 14.04) in a chroot. This allows you to have a ROS Indigo installation even on Ubuntu versions that Indigo doesn't build on.
32. [Installing ROS Indigo in a chroot](/th/ROS/Tutorials/InstallingIndigoInChroot "/th/ROS/Tutorials/InstallingIndigoInChroot")This tutorial walks you through installing ROS Indigo (and Ubuntu 14.04) in a chroot. This allows you to have a ROS Indigo installation even on Ubuntu versions that Indigo doesn't build on.
33. [Installing ROS Indigo in a chroot](/ROS/Tutorials/InstallingIndigoInChroot "/ROS/Tutorials/InstallingIndigoInChroot")This tutorial walks you through installing ROS Indigo (and Ubuntu 14.04) in a chroot. This allows you to have a ROS Indigo installation even on Ubuntu versions that Indigo doesn't build on.
34. [No Title](/ROS/Tutorials/Auto "/ROS/Tutorials/Auto")No Description
35. [No Title](/pt_BR/ROS/Tutorials/Auto "/pt_BR/ROS/Tutorials/Auto")No Description

Wiki: ROS/Tutorials/Auto (last edited 2011-08-22 18:03:30 by [KenConley](/KenConley "KenConley @ fw1-b.willowgarage.com[157.22.19.17]"))

Except where otherwise noted, the ROS wiki is licensed under the   
