import { Component } from '@angular/core';
import {
  MatDialogActions,
  MatDialogClose,
  MatDialogContent,
  MatDialogRef,
  MatDialogTitle,
} from '@angular/material/dialog';
import { inject } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatButtonModule } from '@angular/material/button';
import { FormsModule } from '@angular/forms';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatSelectModule } from '@angular/material/select';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [
    MatDialogContent,
    MatDialogTitle,
    MatDialogActions,
    MatDialogClose,
    MatFormFieldModule,
    MatButtonModule,
    FormsModule,
    MatSlideToggleModule,
    MatSelectModule,
  ],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss',
})
export class SettingsComponent {
  readonly dialogRef = inject(MatDialogRef<SettingsComponent>);
  eval = false;
  embedding = '';
  llm = '';
  readonly http = inject(HttpClient);
  constructor() {
    this.http
      .get('https://8000-01j10v4pkgxq5dx4v24zhwjffh.cloudspaces.litng.ai/get_settings')
      .subscribe((val: any) => {
        this.eval = val['eval'];
        this.embedding = val['embedding'];
        this.llm = val['llm'];
      });
  }
  onNoClick(): void {
    this.dialogRef.close();
  }

  submit() {
    this.http
      .post('https://8000-01j10v4pkgxq5dx4v24zhwjffh.cloudspaces.litng.ai/set_settings', {
        eval: this.eval,
        embedding: this.embedding,
        llm: this.llm,
      })
      .subscribe((val: any) => alert('saved'));
    this.dialogRef.close({});
  }
}
