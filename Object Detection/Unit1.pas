unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants,
  System.Classes, Vcl.Graphics, Math,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.ExtCtrls, Vcl.StdCtrls,
  Vcl.ComCtrls, Vcl.ExtDlgs;

type
  TForm1 = class(TForm)
    Image2: TImage;
    Button2: TButton;
    Edit1: TEdit;
    Label2: TLabel;
    Button1: TButton;
    OpenPictureDialog1: TOpenPictureDialog;
    procedure Button2Click(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

const
  LibraryName = 'tflite.dll';

type
  TfLiteStatus = (kTfLiteOk, kTfLiteError, kTfLiteDelegateError);

  TfLiteType = (kTfLiteNoType = 0, kTfLiteFloat32 = 1, kTfLiteInt32 = 2,
    kTfLiteUInt8 = 3, kTfLiteInt64 = 4, kTfLiteString = 5, kTfLiteBool = 6,
    kTfLiteInt16 = 7, kTfLiteComplex64 = 8, kTfLiteInt8 = 9,
    kTfLiteFloat16 = 10, kTfLiteFloat64 = 11);

function TfLiteModelCreateFromFile(const model_path: PAnsiChar): Pointer;
  stdcall; external LibraryName;

function TfLiteInterpreterCreate(model: Pointer; optional_options: Pointer)
  : Pointer; stdcall; external LibraryName;

function TfLiteInterpreterOptionsCreate(): Pointer; stdcall;
  external LibraryName;

function TfLiteInterpreterAllocateTensors(interpreter: Pointer): TfLiteStatus;
  stdcall; external LibraryName;

function TfLiteInterpreterGetInputTensor(interpreter: Pointer;
  input_index: Int32): Pointer; stdcall; external LibraryName;

function TfLiteInterpreterGetOutputTensor(interpreter: Pointer;
  output_index: Int32): Pointer; stdcall; external LibraryName;

function TfLiteTensorNumDims(tensor: Pointer): Int32; stdcall;
  external LibraryName;

function TfLiteTensorName(tensor: Pointer): PAnsiChar; stdcall;
  external LibraryName;

function TfLiteTensorType(tensor: Pointer): TfLiteType; stdcall;
  external LibraryName;

function TfLiteTensorByteSize(tensor: Pointer): SIZE_T; stdcall;
  external LibraryName;

function TfLiteInterpreterResizeInputTensor(interpreter: Pointer;
  input_index: Int32; input_dims: PInteger; input_dims_size: Int32)
  : TfLiteStatus; stdcall; external LibraryName;

function TfLiteInterpreterGetInputTensorCount(interpreter: Pointer): Int32;
  stdcall; external LibraryName;

function TfLiteInterpreterGetOutputTensorCount(interpreter: Pointer): Int32;
  stdcall; external LibraryName;

function TfLiteTensorCopyFromBuffer(tensor: Pointer; input_data: Pointer;
  input_data_size: SIZE_T): TfLiteStatus; stdcall; external LibraryName;

function TfLiteTensorCopyToBuffer(output_tensor: Pointer; output_data: Pointer;
  output_data_size: SIZE_T): TfLiteStatus; stdcall; external LibraryName;

procedure TfLiteInterpreterOptionsSetNumThreads(options: Pointer;
  num_threads: Int32); stdcall; external LibraryName;

procedure TfLiteInterpreterOptionsDelete(options: Pointer); stdcall;
  external LibraryName;

procedure TfLiteModelDelete(model: Pointer); stdcall; external LibraryName;

function TfLiteInterpreterInvoke(interpreter: Pointer): TfLiteStatus; stdcall;
  external LibraryName;

procedure TForm1.Button1Click(Sender: TObject);
var
  fBitmap: TBitmap;
begin
  if OpenPictureDialog1.Execute then
  begin
    fBitmap := TBitmap.Create;
    try
      fBitmap.LoadFromFile(OpenPictureDialog1.FileName);

      Image2.Picture.Bitmap.Canvas.StretchDraw
        (Image2.Picture.Bitmap.Canvas.ClipRect, fBitmap);
      // Image2.Picture.Bitmap.
    finally
      fBitmap.Free;
    end;

  end;
end;

procedure TForm1.Button2Click(Sender: TObject);
const
  PixelCount = 300 * 300;
type
  PRGBArray = ^TRGBArray;
  TRGBArray = array [0 .. PixelCount - 1] of TRGBTriple;
type
  PInputArray = ^TInputArray;
  TInputArray = array [0 .. 300 * 300 - 1] of array [0 .. 3 - 1] of UInt8;
var
  i, X, Y: DWORD;

  fLibrary: HMODULE;
  fModel: Pointer;
  fInterpreterOptions: Pointer;
  fInterpreter: Pointer;
  fStatus: TfLiteStatus;
  fInputTensorCount, fOutputTensorCount, fNumDims: Int32;
  fInputTensor, fOutputTensor: Pointer;

  fTensorName: PAnsiChar;
  fTensorType: TfLiteType;
  fTensorByteSize: SIZE_T;

  fInputArray: PInputArray;

  fOutputLocations: array [0 .. 10 - 1] of array [0 .. 4 - 1] of Float32;
  fOutputClasses: array [0 .. 10 - 1] of Float32;
  fOutputScores: array [0 .. 10 - 1] of Float32;
  fNumDetections: array [0 .. 1 - 1] of Float32;

  fColors: PRGBArray;
  fLabelMap: TStringList;
  fValue: Extended;
begin
  fLibrary := LoadLibrary(LibraryName);

  if fLibrary = 0 then
  begin
    ShowMessage('Error: Load tensorflow lite library ' + LibraryName + ' - ' +
      SysErrorMessage(GetLastError));
    Exit;
  end;

  try
    fModel := TfLiteModelCreateFromFile(PAnsiChar(AnsiString(Edit1.Text)));

    if fModel = nil then
    begin
      ShowMessage('Error: Create model from file - ' +
        SysErrorMessage(GetLastError));
      Exit;
    end;

    fInterpreterOptions := TfLiteInterpreterOptionsCreate;

    if fInterpreterOptions <> nil then
    begin
      TfLiteInterpreterOptionsSetNumThreads(fInterpreterOptions, 2);

      fInterpreter := TfLiteInterpreterCreate(fModel, fInterpreterOptions);

      // параметры / модель могут быть удалены сразу же после создани¤ интерпретатора
      TfLiteInterpreterOptionsDelete(fInterpreterOptions);
      TfLiteModelDelete(fModel);

      if fInterpreter <> nil then
      begin
        fStatus := TfLiteInterpreterAllocateTensors(fInterpreter);

        fInputTensorCount := TfLiteInterpreterGetInputTensorCount(fInterpreter);
        fOutputTensorCount := TfLiteInterpreterGetOutputTensorCount
          (fInterpreter);

        fInputTensor := TfLiteInterpreterGetInputTensor(fInterpreter, 0);
        fOutputTensor := TfLiteInterpreterGetOutputTensor(fInterpreter, 0);

        if fInputTensor <> nil then
        begin

          // инфа о тензоре
          fNumDims := TfLiteTensorNumDims(fInputTensor);
          fTensorName := TfLiteTensorName(fInputTensor);
          fTensorType := TfLiteTensorType(fInputTensor);
          fTensorByteSize := TfLiteTensorByteSize(fInputTensor);

          // Размер картинки на входе только 300x300

          // Object Detection
          // https://www.tensorflow.org/lite/models/object_detection/overview#get_started

          GetMem(fInputArray, fTensorByteSize);
          try

            // То ли модель такая говнотренированная, то ли фильтр надо сделать средний, то ли я тупой
            for Y := 0 to Image2.Picture.Bitmap.Height - 1 do
            begin
              fColors := PRGBArray(Image2.Picture.Bitmap.ScanLine[Y]);

              for X := 0 to Image2.Picture.Bitmap.Width - 1 do
              begin
                fInputArray[X + (Y * Image2.Picture.Bitmap.Width)][0] :=
                  fColors[X].rgbtRed;
                fInputArray[X + (Y * Image2.Picture.Bitmap.Width)][1] :=
                  fColors[X].rgbtGreen;
                fInputArray[X + (Y * Image2.Picture.Bitmap.Width)][2] :=
                  fColors[X].rgbtBlue;

              end;
            end;

            fStatus := TfLiteTensorCopyFromBuffer(fInputTensor, fInputArray,
              fTensorByteSize);
          finally
            FreeMem(fInputArray, fTensorByteSize);
          end;

          if fStatus = kTfLiteOk then
          begin
            fStatus := TfLiteInterpreterInvoke(fInterpreter);

            if fStatus = kTfLiteOk then
            begin
              // Locations
              fOutputTensor := TfLiteInterpreterGetOutputTensor
                (fInterpreter, 0);

              fTensorByteSize := TfLiteTensorByteSize(fOutputTensor);

              if fOutputTensor <> nil then
              begin
                fStatus := TfLiteTensorCopyToBuffer(fOutputTensor,
                  @fOutputLocations, fTensorByteSize);
              end;

              // Classes
              fLabelMap := TStringList.Create;
              try
                fLabelMap.LoadFromFile('labelmap.txt');

                fOutputTensor := TfLiteInterpreterGetOutputTensor
                  (fInterpreter, 1);

                fTensorByteSize := TfLiteTensorByteSize(fOutputTensor);

                if fOutputTensor <> nil then
                begin
                  fStatus := TfLiteTensorCopyToBuffer(fOutputTensor,
                    @fOutputClasses, fTensorByteSize);
                end;

                // Scores
                fOutputTensor := TfLiteInterpreterGetOutputTensor
                  (fInterpreter, 2);

                fTensorByteSize := TfLiteTensorByteSize(fOutputTensor);

                if fOutputTensor <> nil then
                begin
                  fStatus := TfLiteTensorCopyToBuffer(fOutputTensor,
                    @fOutputScores, fTensorByteSize);
                end;

                // Всегда 10 обьектов даёт, не нужды делать fnumDetections
                // fStatus := TfLiteTensorCopyToBuffer(fOutputTensor,
                // @fnumDetections, fTensorByteSize);

                // There will always be 10 objects detected
                { 0	Locations	Multidimensional array of [10][4] floating point values between 0 and 1, the inner arrays representing bounding boxes in the form [top, left, bottom, right]
                  1	Classes	Array of 10 integers (output as floating point values) each indicating the index of a class label from the labels file
                  2	Scores	Array of 10 floating point values between 0 and 1 representing probability that a class was detected
                  3	Number and detections }

                if fStatus = kTfLiteOk then
                begin
                  for i := 0 to Length(fOutputLocations[0]) - 1 do
                  begin
                    Image2.Canvas.Brush.Style := bsClear;
                    Image2.Canvas.FillRect(Image2.Canvas.ClipRect);
                    Image2.Canvas.Pen.Color := clRed;

                    // И опять гениальнейшее мое решение
                    fValue := StrToFloat
                      (Copy(FloatToStr(fOutputScores[i]), 1, 4));

                    if fValue >= 0.6 then
                    begin
                      Image2.Canvas.Rectangle
                        (Round(fOutputLocations[i][1] * 300),
                        Round(fOutputLocations[i][0] * 300),
                        Round(fOutputLocations[i][3] * 300),
                        Round(fOutputLocations[i][2] * 300));

                      Image2.Canvas.Brush.Style := bsSolid;

                      Image2.Canvas.Brush.Color := clRed;
                      Image2.Canvas.Font.Color := clWhite;

                      // SSD Mobilenet V1 Model assumes class 0 is background class
                      // in label file and class labels start from 1 to number_of_classes+1,
                      // while outputClasses correspond to class index from 0 to number_of_classes
                      // i начанаеться с 0 поэтому + 1
                      Image2.Canvas.TextOut(Round(fOutputLocations[i][1] * 300)
                        + 2, Round(fOutputLocations[i][0] * 300) + 1,
                        fLabelMap.Strings[Round(fOutputClasses[i]) + 1] + ' - '
                        + FloatToStr(fValue));

                    end;
                  end;

                  Beep;

                end;
              finally
                fLabelMap.Free;
              end;
            end;
          end;
        end;
      end;
    end;
  finally
    FreeLibrary(fLibrary);
  end;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  OpenPictureDialog1.InitialDir := GetCurrentDir;
end;

end.
