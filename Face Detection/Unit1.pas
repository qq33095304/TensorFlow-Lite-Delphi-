unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants,
  System.Classes, Vcl.Graphics, Math,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.ExtCtrls, Vcl.StdCtrls,
  Vcl.ComCtrls, Vcl.ExtDlgs, Vcl.Imaging.jpeg;

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
  TInputArray = array [0 .. 300 - 1] of array [0 .. 300 - 1] of array
    [0 .. 3 - 1] of Float32;
type
  PDetectionBoxes = ^TDetectionBoxes;
  TDetectionBoxes = array [0 .. 10 - 1] of array [0 .. 4 - 1] of Float32;
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
  fDetectionBoxes: PDetectionBoxes;
  fDetectionScores: array [0 .. 1 - 1] of array [0 .. 10 - 1] of Float32;

  fColors: PRGBArray;
  fValue: Extended;
begin
  { Сетка ssd_mobilenet_v1_coco_11_06_2017 тренировалась на 350 картинках на крупных китайских фейсах, примерно 4 часа
    время детектирования на CPU - AMD Ryzen 3 2200 занимает около 0.2 сек., медленно
    надо тренировать ssd mobilenet v2, ssd mobilenet v3 и проверить на GPU на андроиде }

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

        // fInputTensorCount := TfLiteInterpreterGetInputTensorCount(fInterpreter);
        // fOutputTensorCount := TfLiteInterpreterGetOutputTensorCount(fInterpreter);

        fInputTensor := TfLiteInterpreterGetInputTensor(fInterpreter, 0);
        fOutputTensor := TfLiteInterpreterGetOutputTensor(fInterpreter, 0);

        if fInputTensor <> nil then
        begin
          // инфа о тензоре
          // fNumDims := TfLiteTensorNumDims(fInputTensor);
          // fTensorName := TfLiteTensorName(fInputTensor);
          // fTensorType := TfLiteTensorType(fInputTensor);
          fTensorByteSize := TfLiteTensorByteSize(fInputTensor);
          // fTensorByteSize := SizeOf(TInputArray);

          // 'normalized_input_image_tensor': a float32 tensor of shape
          // [1, height, width, 3] containing the normalized input image.

          GetMem(fInputArray, fTensorByteSize);
          try
            { In floating point Mobilenet model, 'normalized_image_tensor' has values
              between [-1,1). This typically means mapping each pixel (linearly)
              to a value between [-1, 1]. Input image
              values between 0 and 255 are scaled by (1/128.0) and then a value of
              -1 is added to them to ensure the range is [-1,1). }

            for Y := 0 to Image2.Picture.Bitmap.Height - 1 do
            begin
              fColors := PRGBArray(Image2.Picture.Bitmap.ScanLine[Y]);

              for X := 0 to Image2.Picture.Bitmap.Width - 1 do
              begin
                fInputArray[Y][X][0] := (fColors[X].rgbtRed * 0.0078125) - 1;
                fInputArray[Y][X][1] := (fColors[X].rgbtGreen * 0.0078125) - 1;
                fInputArray[Y][X][2] := (fColors[X].rgbtBlue * 0.0078125) - 1;
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
              { Номер тензора в Output
                0:detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with boxlocations
                1:detection_classes: a float32 tensor of shape [1, num_boxes] with class indices
                2:detection_scores: a float32 tensor of shape [1, num_boxes] with class scores
                3:num_boxes: a float32 tensor of size 1 containing the number of detected boxes }

              // Не нужно выделять память
              // detection_scores
              fOutputTensor := TfLiteInterpreterGetOutputTensor
                (fInterpreter, 2);

              if fOutputTensor <> nil then
              begin
                fTensorByteSize := TfLiteTensorByteSize(fOutputTensor);

                TfLiteTensorCopyToBuffer(fOutputTensor, @fDetectionScores,
                  fTensorByteSize);
              end;

              // detection_boxes
              fOutputTensor := TfLiteInterpreterGetOutputTensor
                (fInterpreter, 0);

              if fOutputTensor <> nil then
              begin
                // Выделяем память
                fTensorByteSize := TfLiteTensorByteSize(fOutputTensor);

                GetMem(fDetectionBoxes, fTensorByteSize);
                try
                  fStatus := TfLiteTensorCopyToBuffer(fOutputTensor,
                    fDetectionBoxes, fTensorByteSize);

                  if fStatus = kTfLiteOk then
                  begin
                    for i := 0 to (10) - 1 do
                    begin
                      Image2.Canvas.Brush.Style := bsClear;
                      Image2.Canvas.FillRect(Image2.Canvas.ClipRect);
                      Image2.Canvas.Pen.Color := clRed;

                      // Гениальнейшее решение
                      fValue := StrToFloat
                        (Copy(FloatToStr(fDetectionScores[0][i]), 1, 4));

                      if fValue >= 0.6 then
                      begin
                        { Left := fDetectionBoxes[i][1]
                          Top := fDetectionBoxes[i][0]
                          Right := fDetectionBoxes[i][3]
                          Bottom := fDetectionBoxes[i][2] }

                        Image2.Canvas.Rectangle
                          (Round(fDetectionBoxes[i][1] * 300),
                          Round(fDetectionBoxes[i][0] * 300),
                          Round(fDetectionBoxes[i][3] * 300),
                          Round(fDetectionBoxes[i][2] * 300));

                        Image2.Canvas.Brush.Style := bsSolid;
                        Image2.Canvas.Brush.Color := clRed;
                        Image2.Canvas.Font.Color := clWhite;

                        Image2.Canvas.TextOut(Round(fDetectionBoxes[i][1] * 300)
                          + 2, Round(fDetectionBoxes[i][0] * 300) + 1,
                          FloatToStr(fValue));
                      end;
                    end;

                    Beep;

                  end;
                finally
                  FreeMem(fDetectionBoxes, fTensorByteSize);
                end;
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
