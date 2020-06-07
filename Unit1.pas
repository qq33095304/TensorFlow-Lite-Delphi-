unit Unit1;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants,
  System.Classes, Vcl.Graphics, Math,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.ExtCtrls, Vcl.StdCtrls,
  Vcl.ComCtrls;

type
  TForm1 = class(TForm)
    Image1: TImage;
    Image2: TImage;
    Button1: TButton;
    Button2: TButton;
    ListView1: TListView;
    Label1: TLabel;
    Edit1: TEdit;
    Label2: TLabel;
    procedure Image2MouseMove(Sender: TObject; Shift: TShiftState;
      X, Y: Integer);
    procedure Image2MouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure Image2MouseUp(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure ListView1Compare(Sender: TObject; Item1, Item2: TListItem;
      Data: Integer; var Compare: Integer);
    procedure FormShow(Sender: TObject);
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
begin
  Image2.Canvas.Brush.Style := bsSolid;
  Image2.Canvas.Brush.Color := clBlack;
  Image2.Canvas.FillRect(Image2.Canvas.ClipRect);
  Image2.Canvas.Pen.Color := clWhite;
end;

procedure TForm1.Button2Click(Sender: TObject);
var
  i, X, Y: DWORD;

  fLibrary: HMODULE;
  fModel: Pointer;
  fInterpreterOptions: Pointer;
  fInterpreter: Pointer;
  fStatus: TfLiteStatus;
  fInputTensorCount, fOutputTensorCount, fNumDims: Int32;
  fInputTensor, fOutputTensor: Pointer;
  fInputDims: Integer;
  fTensorName: PAnsiChar;
  fTensorType: TfLiteType;
  fTensorByteSize: SIZE_T;

  // размеры входного изображнения mnist 28X28 пикселей
  fInput: array [0 .. 28 * 28 - 1] of Float32;
  // Output это массив confidence, 10 цифр  0 .. 9
  fOutput: array [0 .. 10 - 1] of Float32;

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

      // Параметры / модель могут быть удалены сразу же после создания интерпретатора
      TfLiteInterpreterOptionsDelete(fInterpreterOptions);
      TfLiteModelDelete(fModel);

      if fInterpreter <> nil then
      begin
        fStatus := TfLiteInterpreterAllocateTensors(fInterpreter);

        fInputTensorCount := TfLiteInterpreterGetInputTensorCount(fInterpreter);
        fOutputTensorCount := TfLiteInterpreterGetOutputTensorCount
          (fInterpreter);

        // fLiteTensor* TfLiteInterpreterGetInputTensor(const TfLiteInterpreter* interpreter, int32_t input_index);
        // возвращаемая структура TfLiteTensor
        // пример простой, нет нужды всЁ переводить из це
        // вообще исходники тута https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c
        { typedef struct TfLiteTensor {
          TfLiteType type;
          TfLitePtrUnion data;
          TfLiteIntArray* dims;
          TfLiteQuantizationParams params;
          TfLiteAllocationType allocation_type;
          size_t bytes;
          const void* allocation;
          const char* name;
          struct TfLiteDelegate* delegate;
          TfLiteBufferHandle buffer_handle;
          bool data_is_stale;
          bool is_variable;
          TfLiteQuantization quantization;
          TfLiteSparsity* sparsity;
          const TfLiteIntArray* dims_signature; }
        fInputTensor := TfLiteInterpreterGetInputTensor(fInterpreter, 0);
        fOutputTensor := TfLiteInterpreterGetOutputTensor(fInterpreter, 0);

        if fInputTensor <> nil then
        begin
          // Инфа о тензоре
          // fNumDims := TfLiteTensorNumDims(fInputTensor);
          // fTensorName := TfLiteTensorName(fInputTensor);
          // fTensorType := TfLiteTensorType(fInputTensor);
          fTensorByteSize := TfLiteTensorByteSize(fInputTensor);

          // записываем пикселя в fInput,  сверху вниз, слева направо
          for Y := 0 to Image1.Picture.Bitmap.Height - 1 do
          begin
            for X := 0 to Image1.Picture.Bitmap.Width - 1 do
            begin
              if (Image1.Canvas.Pixels[X, Y] > 0) then
                fInput[X + (Y * Image1.Picture.Bitmap.Width)] := 1
              else
                fInput[X + (Y * Image1.Picture.Bitmap.Width)] := 0;
            end;
          end;

          // fTensorByteSize = Length(fInput) * SizeOf(Float32)
          fStatus := TfLiteTensorCopyFromBuffer(fInputTensor, @fInput,
            fTensorByteSize);

          fStatus := TfLiteInterpreterInvoke(fInterpreter);

          if fStatus = kTfLiteOk then
          begin
            for i := 0 to High(fOutput) do
              fOutput[i] := 0;

            fOutputTensor := TfLiteInterpreterGetOutputTensor(fInterpreter, 0);

            // Инфа о тензоре
            // fNumDims := TfLiteTensorNumDims(fOutputTensor);
            // fTensorName := TfLiteTensorName(fOutputTensor);
            // fTensorType := TfLiteTensorType(fOutputTensor);
            fTensorByteSize := TfLiteTensorByteSize(fOutputTensor);

            if fOutputTensor <> nil then
            begin
              // fTensorByteSize = Length(fOutput) * SizeOf(Float32)
              fStatus := TfLiteTensorCopyToBuffer(fOutputTensor, @fOutput,
                fTensorByteSize);

              if fStatus = kTfLiteOk then
              begin
                ListView1.Items.Clear;

                for i := 0 to Length(fOutput) - 1 do
                begin
                  // гениальное решение, крутейшая конвертация
                  fValue := StrToFloat(Copy(FloatToStr(fOutput[i]), 1, 17));

                  if fValue <= 1 then
                  begin
                    with ListView1.Items.Add do
                    begin
                      Caption := FloatToStrF(fValue, ffNumber, 17, 17);
                      SubItems.Add(IntToStr(i));
                    end;
                  end;
                end;

                ListView1.AlphaSort;

                Beep;
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

procedure TForm1.FormShow(Sender: TObject);
begin
  Image2.Canvas.Brush.Style := bsSolid;
  Image2.Canvas.Brush.Color := clBlack;
  Image2.Canvas.FillRect(Image2.Canvas.ClipRect);
end;

procedure TForm1.ListView1Compare(Sender: TObject; Item1, Item2: TListItem;
  Data: Integer; var Compare: Integer);
begin
  Compare := CompareText(Item2.Caption, Item1.Caption);
end;

var
  IsDrawing: Boolean;

procedure TForm1.Image2MouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  case Button of
    mbLeft:
      begin
        IsDrawing := True;

        Image2.Canvas.Pen.Color := clWhite;
        Image2.Canvas.Pen.Width := 28;
        Image2.Canvas.MoveTo(X, Y);
        Image2.Canvas.LineTo(X, Y);
      end;
  end;
end;

procedure TForm1.Image2MouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: Integer);
begin
  if IsDrawing then
  begin
    Image2.Canvas.MoveTo(X, Y);
    Image2.Canvas.LineTo(X, Y);
  end;
end;

procedure TForm1.Image2MouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  IsDrawing := False;

  Image1.Canvas.StretchDraw(Image1.Canvas.ClipRect, Image2.Picture.Bitmap);

  Button2.Click;
end;

end.
