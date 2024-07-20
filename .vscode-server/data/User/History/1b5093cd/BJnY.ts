import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.scss',
})
export class UploadComponent {
  fileName = '';
  loading = false;

  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    this.loading = true;
    const file: File = event.target.files[0];

    if (file) {
      this.fileName = file.name;

      const formData = new FormData();

      formData.append('file', file);

      const upload$ = this.http.post(
        'http://localhost:8000/Upload_File',
        formData
      );

      upload$.subscribe((res: any) => {
        this.loading = false;
      });
    }
  }
}
